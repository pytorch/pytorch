import argparse
import csv
import sys
from pathlib import Path


# Add parent directory to path for imports
sys.path.append(str(Path(__file__).absolute().parents[1]))
sys.path.append(
    str(
        Path(__file__).absolute().parents[3]
        / "benchmarks"
        / "dynamo"
        / "microbenchmarks"
    )
)

from operator_inp_utils import (  # type: ignore[import-not-found]
    deserialize_args,
    OperatorInputsLoader,
)

import torch
from torch._inductor.fx_passes.pad_mm import (
    get_alignment_size_dtype,  # type: ignore[import-not-found]
)
from torch._subclasses.fake_tensor import FakeTensorMode


def is_aligned(dim: int, align_size: int) -> bool:
    """Check if dimension is aligned to the given alignment size."""
    return dim % align_size == 0


def extract_mm_shapes_from_loader(
    loader: OperatorInputsLoader,
) -> list[tuple[int, int, int, torch.dtype, torch.dtype]]:
    """Extract matrix multiplication shapes from an OperatorInputsLoader using deserialize_args with FakeTensorMode."""
    shapes = []

    # Matrix multiplication operators to look for
    mm_operators = ["aten.mm.default", "aten.addmm.default", "aten.bmm.default"]

    # Use FakeTensorMode to avoid instantiating actual tensors
    with FakeTensorMode():
        for op_name in mm_operators:
            if op_name not in loader.operator_db:
                continue

            # Count shapes extracted from this operator
            shape_count = 0

            # Access the raw string data directly from operator_db and reuse existing parsing
            for input_str in loader.operator_db[op_name]:
                try:
                    # Use deserialize_args to parse inputs - will create fake tensors
                    args, kwargs = deserialize_args(input_str)

                    if op_name == "aten.mm.default":
                        # mm(input, mat2) -> result
                        if len(args) >= 2:
                            a, b = args[0], args[1]
                            if isinstance(a, torch.Tensor) and isinstance(
                                b, torch.Tensor
                            ):
                                a_shape, a_dtype = tuple(a.shape), a.dtype
                                b_shape, b_dtype = tuple(b.shape), b.dtype
                                if len(a_shape) == 2 and len(b_shape) == 2:
                                    m, k = a_shape
                                    k2, n = b_shape
                                    if k == k2:  # Valid matrix multiplication
                                        shapes.append((m, k, n, a_dtype, b_dtype))
                                        shape_count += 1

                    elif op_name == "aten.addmm.default":
                        # addmm(bias, input, mat2) -> result
                        if len(args) >= 3:
                            _, a, b = args[0], args[1], args[2]
                            if isinstance(a, torch.Tensor) and isinstance(
                                b, torch.Tensor
                            ):
                                a_shape, a_dtype = tuple(a.shape), a.dtype
                                b_shape, b_dtype = tuple(b.shape), b.dtype
                                if len(a_shape) == 2 and len(b_shape) == 2:
                                    m, k = a_shape
                                    k2, n = b_shape
                                    if k == k2:  # Valid matrix multiplication
                                        shapes.append((m, k, n, a_dtype, b_dtype))
                                        shape_count += 1

                    elif op_name == "aten.bmm.default":
                        # bmm(input, mat2) -> result (batch matrix multiplication)
                        if len(args) >= 2:
                            a, b = args[0], args[1]
                            if isinstance(a, torch.Tensor) and isinstance(
                                b, torch.Tensor
                            ):
                                a_shape, a_dtype = tuple(a.shape), a.dtype
                                b_shape, b_dtype = tuple(b.shape), b.dtype
                                if len(a_shape) == 3 and len(b_shape) == 3:
                                    batch1, m, k = a_shape
                                    batch2, k2, n = b_shape
                                    if (
                                        batch1 == batch2 and k == k2
                                    ):  # Valid batch matrix multiplication
                                        shapes.append((m, k, n, a_dtype, b_dtype))
                                        shape_count += 1

                except Exception:
                    # Skip invalid inputs
                    continue

            print(f"    Extracted {shape_count} shapes from {op_name}")

    return shapes


def filter_unaligned_shapes(
    shapes: list[tuple[int, int, int, torch.dtype, torch.dtype]],
) -> list[tuple[int, int, int, torch.dtype, torch.dtype]]:
    """Filter shapes to keep only those that are not completely aligned (so padding is relevant)."""
    filtered_shapes = []

    for m, k, n, dtype1, dtype2 in shapes:
        # Use the primary dtype for alignment calculation (assume both dtypes are similar for alignment purposes)
        dtype = dtype1
        try:
            align_size = get_alignment_size_dtype(dtype)

            # Only keep shapes where not all dimensions are aligned
            if not all(is_aligned(dim, align_size) for dim in [m, k, n]):
                filtered_shapes.append((m, k, n, dtype1, dtype2))

        except Exception:
            # If we can't get alignment size, skip this shape
            continue

    return filtered_shapes


def collect_known_mm_shapes() -> list[tuple[int, int, int, torch.dtype, torch.dtype]]:
    """
    Collect known matrix multiplication shapes from HuggingFace, TIMM, and TorchBench datasets.

    Returns:
        List of tuples containing (m, k, n, dtype1, dtype2) for matrix multiplication shapes
        that are not completely aligned (so padding is relevant).
    """
    all_shapes = []

    loaders = []

    # Try to load each dataset
    try:
        hf_loader = OperatorInputsLoader.get_huggingface_loader()
        loaders.append(("HuggingFace", hf_loader))
    except Exception as e:
        print(f"Warning: Could not load HuggingFace dataset: {e}")

    try:
        timm_loader = OperatorInputsLoader.get_timm_loader()
        loaders.append(("TIMM", timm_loader))
    except Exception as e:
        print(f"Warning: Could not load TIMM dataset: {e}")

    try:
        torchbench_loader = OperatorInputsLoader.get_torchbench_loader()
        loaders.append(("TorchBench", torchbench_loader))
    except Exception as e:
        print(f"Warning: Could not load TorchBench dataset: {e}")

    # Extract shapes from each loader
    for dataset_name, loader in loaders:
        print(f"Extracting shapes from {dataset_name}...")

        shapes = extract_mm_shapes_from_loader(loader)
        print(f"Found {len(shapes)} matrix multiplication shapes from {dataset_name}")
        all_shapes.extend(shapes)

    # Remove duplicates
    unique_shapes = list(set(all_shapes))
    print(f"Total unique shapes before filtering: {len(unique_shapes)}")

    # Filter for unaligned shapes only
    filtered_shapes = filter_unaligned_shapes(unique_shapes)
    print(f"Shapes after filtering for unaligned: {len(filtered_shapes)}")

    return filtered_shapes


def main(output_file="mm_shapes.csv"):
    shapes = collect_known_mm_shapes()

    print(f"\nCollected {len(shapes)} real-world matrix multiplication shapes")

    # Convert dtype objects to strings and filter for desired dtypes
    dtype_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }

    # Convert to desired format and filter dtypes
    csv_rows = []
    for m, k, n, dtype1, dtype2 in shapes:
        # Use the first dtype and convert to string
        if dtype1 in dtype_map:
            dtype_str = dtype_map[dtype1]
            csv_rows.append([m, k, n, dtype_str])

    # Save to CSV file
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(["M", "K", "N", "dtype"])
        # Write data rows
        writer.writerows(csv_rows)

    print(f"Saved matrix multiplication shapes to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect matrix multiplication shapes from real-world datasets and save to CSV"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="mm_shapes.csv",
        help="Output CSV filename (default: mm_shapes.csv)",
    )

    args = parser.parse_args()
    main(args.output)
