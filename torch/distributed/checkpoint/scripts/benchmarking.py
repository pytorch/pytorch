# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

"""
This script demonstrates how to:
1. Load tensors
2. Split them into d-tensors (distributed tensors)
3. Save them using DCP (Distributed Checkpoint) with HuggingFace integration
4. Run the consolidation function to consolidate the sharded files back to full tensors
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import consolidate_safetensors_files
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard


def init_distributed():
    """Initialize the distributed environment."""
    if not dist.is_available() or not dist.is_initialized():
        if torch.cuda.is_available():
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")

        # Set device to current rank
        if torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank())

    return dist.get_world_size(), dist.get_rank()


def load_and_split_tensors(model_path: str, device_mesh):
    """
    Load tensors from a model and split them into d-tensors.

    Args:
        model_path: Path to the model to load
        device_mesh: Device mesh to distribute tensors on

    Returns:
        state_dict with distributed tensors
    """
    # Load the model
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_path)

    # Create distributed tensors
    distributed_state_dict = {}
    for key, tensor in state_dict.items():
        # Choose sharding strategy based on tensor size
        if tensor.dim() >= 2 and tensor.size(0) > 10:
            # Shard along the first dimension for large tensors
            placements = [Shard(0)]
        else:
            # Replicate small tensors
            placements = [Replicate()]

        # Distribute the tensor
        dtensor = distribute_tensor(tensor, device_mesh, placements=placements)
        distributed_state_dict[key] = dtensor

    return distributed_state_dict, state_dict


def save_with_dcp_huggingface(state_dict, output_dir: str):
    """
    Save the distributed state dict using DCP with HuggingFace integration.

    Args:
        state_dict: State dict with distributed tensors
        output_dir: Directory to save the checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the distributed checkpoint
    dist_cp.save(
        state_dict=state_dict,
        storage_writer=dist_cp._HuggingFaceStorageWriter(
            path=output_dir,
            save_sharded=True,  # Save as sharded files
        ),
    )

    print(f"Saved distributed checkpoint to {output_dir}")


def save_safetensors(state_dict, output_dir: str):
    """
    Save the state dict as safetensors.

    Args:
        state_dict: State dict with tensors
        output_dir: Directory to save the checkpoint
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save the checkpoint
    from safetensors.torch import save_file

    save_file(state_dict, os.path.join(output_dir, "checkpoint.safetensors"))


def consolidate_checkpoint(input_dir: str, output_dir: str, fqn_to_index_mapping=None):
    """
    Consolidate the sharded checkpoint files into full tensors.

    Args:
        input_dir: Directory containing sharded checkpoint files
        output_dir: Directory to save the consolidated checkpoint
        fqn_to_index_mapping: Optional mapping of tensor names to output file indices
    """
    os.makedirs(output_dir, exist_ok=True)

    # Consolidate the checkpoint
    consolidate_safetensors_files(
        input_dir=input_dir,
        output_dir=output_dir,
        fqn_to_index_mapping=fqn_to_index_mapping,
    )

    print(f"Consolidated checkpoint saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Split tensors into d-tensors, save with DCP HuggingFace, and consolidate"
    )
    """
    parser.add_argument(
        "--model-path", type=str, default="", help="Path to the model to load"
    )
    parser.add_argument(
        "--sharded-output-dir",
        type=str,
        default="./sharded_checkpoint",
        help="Directory to save the sharded checkpoint",
    )
    parser.add_argument(
        "--consolidated-output-dir",
        type=str,
        default="./consolidated_checkpoint",
        help="Directory to save the consolidated checkpoint",
    )
    args = parser.parse_args()
    """

    # Initialize distributed environment
    world_size, rank = init_distributed()
    print(f"Running with world_size={world_size}, rank={rank}")

    model_path = "/home/ankitageorge/.cache/huggingface/hub/models--meta-llama--Llama-3.3-70B-Instruct/snapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b/model-00001-of-00030.safetensors"
    base_dir = "/data/users/ankitageorge/testing/"
    sharded_output_dir = os.path.join(base_dir, "sharded_checkpoint")
    dcp_consolidated_output_dir = os.path.join(base_dir, "dcp_consolidated_checkpoint")
    consolidated_output_dir = os.path.join(base_dir, "consolidated_checkpoint")

    # Initialize device mesh
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    device_mesh = init_device_mesh(device_type, (world_size,))

    # Load and split tensors
    distributed_state_dict, state_dict = load_and_split_tensors(model_path, device_mesh)
    start_time = time.time()
    if rank == 0:
        save_safetensors(state_dict, consolidated_output_dir)
        print("time to save as safetensors ", time.time() - start_time)

    # Save with DCP HuggingFace
    start_time = time.time()
    save_with_dcp_huggingface(distributed_state_dict, sharded_output_dir)

    # Make sure all processes have finished saving
    dist.barrier()
    print("time to save with DCP HuggingFace ", time.time() - start_time)

    # Only rank 0 needs to consolidate the checkpoint
    start_time = time.time()
    if rank == 0:
        consolidate_checkpoint(sharded_output_dir, dcp_consolidated_output_dir)

        print("Time to consolidate checkpoint: ", time.time() - start_time)


if __name__ == "__main__":
    main()
