import subprocess
import os
from typing import Dict
from vllm_build import build_vllm
import argparse


def build():
    parser = argparse.ArgumentParser(description="Build docker images for various targets")
    parser.add_argument("--target", required=True, help="Target to build (e.g., vllm)")

    args = parser.parse_args()
    target = args.target

    match target:
        case "vllm":
            build_vllm()
        case _:
            print(f"Unknown target: {target}")
            return
if __name__ == "__main__":
    build()
