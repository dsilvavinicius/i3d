#!/usr/bin/env python
# coding: utf-8

"""
Simple script to run multiple SDF reconstructions given a base log directory
and a set of checkpoints.
"""


import argparse
import json
import shlex
import shutil
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(
        description="Runs the reconstruction tests."
    )
    parser.add_argument(
        "base_dir",
        help="Checkpoint base directory. e.g. ./logs/test1"
    )
    parser.add_argument(
        "--checkpoints", "-c", nargs="+", default=["final"],
        help="Checkpoints to use when reconstructing the model."
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=256,
        help="Marching cubes resolution"
    )
    parser.add_argument('--w0', "-w0", type=int, default=30,
               help='Multiplicative factor for the frequencies')

    args = parser.parse_args()

    output_dir = os.path.join(args.base_dir, "reconstructions")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        print("[WARN] Output path exists, overwritting contents.")

    params = dict()


    script_meshing = "test_two_sdfs"

    checkpoint_dir = os.path.join(args.base_dir, "checkpoints")

    for c in args.checkpoints:

        experiment_name = args.base_dir.split("/")[-1] + f"_checkpoint-{c}"
        default_args = f"python experiment_scripts/{script_meshing}.py --experiment_name={experiment_name} --w0={args.w0}"

        model_name = ""
        if c == "final":
            model_name = "model_final.pth"
        elif c == "current":
            model_name = "model_current.pth"
        else:
            model_name = f"model_epoch_{c:0>4}.pth"

        checkpoint_path = checkpoint_dir + "/" + model_name
        checkpoint_args = f"{default_args} --checkpoint_path={checkpoint_path}"

        params[c] = dict()

        resolution_divisor = 1
        while args.resolution // resolution_divisor > 1:
            r = args.resolution // resolution_divisor
            params[c]["resolution"] = r
            cmd_line = f"{checkpoint_args} --resolution={r}"

            lex_args = shlex.split(cmd_line)
            try:
                subprocess.run(lex_args, check=True)
            except subprocess.CalledProcessError:
                print(f"[WARN] Error when calling {script_meshing}.py with args: \"{lex_args}\"")
                resolution_divisor *= 2
                continue
            else:
                break

        shutil.copyfile(
            os.path.join("logs", experiment_name, "test.ply"),
            os.path.join(output_dir, f"{c}.ply")
        )

        shutil.rmtree(os.path.join("logs", experiment_name))

        with open(os.path.join(args.base_dir, "reconstruction_params.json"), "w+") as fout:
            json.dump(params, fout, sort_keys=True, indent=4)


if __name__ == "__main__":
    main()
