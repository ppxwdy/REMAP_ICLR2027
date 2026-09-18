#!/usr/bin/env python3
"""
ReMAP experiment driver (residual / gnn_only / free ablations).

Examples:
  python exp_remap.py --mode residual --seed 1211
  python exp_remap.py --mode gnn_only --seed 66 --rounding argmax
  python exp_remap.py --mode free --seed 88 --rounding ce
"""
import argparse
import json
import os
from datetime import datetime
from time import time

import pandas as pd

from GG_net_remap import *
from utils2 import *


def parse_args():
    parser = argparse.ArgumentParser(description="ReMAP unsupervised GNN experiments")
    parser.add_argument(
        "--mode",
        type=str,
        default="residual",
        choices=["gnn_only", "free", "residual"],
        help="Logits parameterization: gnn_only | free | residual (u + rho_v(G))",
    )
    parser.add_argument("--seed", type=int, default=1211)
    parser.add_argument(
        "--data-path",
        type=str,
        default="",
    )
    parser.add_argument("--net-key", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--dim-embedding", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=1024, help="GNN hidden width (nerouns)")
    parser.add_argument("--num-layers", type=int, default=5, help="Number of GraphSAGE layers")
    parser.add_argument("--num-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--rounding",
        type=str,
        default="ce",
        choices=["ce", "argmax"],
        help="Incumbent decoder: ce (conditional-expectation) or argmax",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--temperature-schedule",
        type=str,
        default="constant",
        help='JSON string, e.g. \'{"type":"linear","start":2.0,"end":1.0,"steps":200}\'',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:6" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="result_files/remap",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds; if set, overrides --seed",
    )
    return parser.parse_args()


def parse_temperature_schedule(schedule_str):
    if schedule_str is None or schedule_str.lower() in ("none", "constant", ""):
        return None
    try:
        return json.loads(schedule_str)
    except json.JSONDecodeError:
        return {"type": "constant", "value": float(schedule_str)}


def main():
    args = parse_args()
    temperature_schedule = parse_temperature_schedule(args.temperature_schedule)

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    else:
        seeds = [args.seed]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"{args.mode}-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Results will be saved to: {os.path.abspath(run_dir)}")

    torch_device = torch.device(args.device)
    torch_dtype = torch.float

    files = os.listdir(args.data_path)
    all_metrics = []

    for seed in seeds:
        setup_seed(seed)
        print(f"\n{'=' * 80}\nseed={seed}  mode={args.mode}  net_key={args.net_key}\n{'=' * 80}")

        for file in sorted(files):
            if not file.endswith(".uai"):
                continue

            nodes, edges, cliques, states, unary_energy_p, pair_energy_p, max_state, multi, zero_e = (
                read_clique_model2(os.path.join(args.data_path, file))
            )
            print(file, len(nodes), len(cliques))

            nodes_num = len(nodes)
            gnn_hypers = {
                "num_features": args.dim_embedding,
                "num_classes": max_state,
                "nerouns": args.hidden_dim,
                "dropout": 0.0,
                "num_layers": args.num_layers,
            }
            opt_params = {"lr": args.lr}

            net, optimizer, embed = create_net(
                nodes_num,
                gnn_hypers,
                opt_params,
                torch_device,
                torch_dtype,
                net_type=args.net_key,
                mode=args.mode,
            )

            label_mask = create_variable_class_mask(nodes_num, states, torch_device)
            net.mask = label_mask
            net.train()

            t0 = time()
            loss_curve, assignment, metrics = train(
                nodes,
                edges,
                cliques,
                unary_energy_p,
                pair_energy_p,
                label_mask,
                net,
                args.num_iter,
                optimizer,
                embed,
                torch_device,
                args.dim_embedding,
                seed=seed,
                mode=args.mode,
                temperature=args.temperature,
                temperature_schedule=temperature_schedule,
                rounding=args.rounding,
            )
            metrics["file"] = file
            metrics["num_nodes"] = nodes_num
            metrics["num_cliques"] = len(cliques)
            metrics["net_key"] = args.net_key
            metrics["dim_embedding"] = args.dim_embedding
            metrics["hidden_dim"] = args.hidden_dim
            metrics["num_layers"] = args.num_layers
            metrics["lr"] = args.lr
            metrics["rounding"] = args.rounding
            metrics["total_wall_clock_including_io"] = time() - t0
            all_metrics.append(metrics)

            if loss_curve is not None:
                curve_path = os.path.join(
                    run_dir,
                    f"{file.replace('.uai', '')}_{args.mode}_seed{seed}_dim{args.dim_embedding}.csv",
                )
                pd.DataFrame({"loss": loss_curve}).to_csv(curve_path, index=False)

            if not isinstance(assignment, bool):
                check_feasi(assignment, states)
            else:
                print("No result!")
            print()

    if all_metrics:
        metrics_path = os.path.join(
            run_dir,
            f"metrics_{args.mode}_net{args.net_key}.csv",
        )
        pd.DataFrame(all_metrics).to_csv(metrics_path, index=False)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
