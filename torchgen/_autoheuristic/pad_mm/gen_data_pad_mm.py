import csv
import random
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any


sys.path.append(str(Path(__file__).absolute().parents[1]))

from benchmark_runner import BenchmarkRunner  # type: ignore[import-not-found]
from benchmark_utils import (  # type: ignore[import-not-found]
    fits_in_memory,
    get_mm_tensors,
    get_random_between_pow2,
    set_precision,
    transpose_tensors,
)
from collect_known_mm_shapes import (
    collect_known_mm_shapes,  # type: ignore[import-not-found]
)

import torch
from torch._inductor.fx_passes.pad_mm import (  # type: ignore[import-not-found]
    get_alignment_size_dtype,
)
from torch._inductor.utils import fresh_cache


class BenchmarkRunnerPadMM(BenchmarkRunner):  # type: ignore[misc, no-any-unimported]
    """
    BenchmarkRunner for pad_mm. Used to generate collect training data with AutoHeuristic to learn a heuristic.
    """

    def __init__(self) -> None:
        super().__init__("pad_mm")

        # Add CLI argument for additional shape CSV files
        self.parser.add_argument(
            "--additional-shape-csv",
            nargs="*",
            default=[],
            help="List of CSV files containing additional matrix multiplication shapes (M,K,N,dtype format)",
        )

        # Initialize additional_shape_collections
        self.additional_shape_collections: list[
            list[tuple[int, int, int, torch.dtype, torch.dtype]]
        ] = []

        # Initialize the shape generator (will be set up after parsing args)
        self.shape_generator = None

    def load_shapes_from_csv(
        self, csv_file: str
    ) -> list[tuple[int, int, int, torch.dtype, torch.dtype]]:
        """Load matrix multiplication shapes from a CSV file in M,K,N,dtype format."""
        shapes = []
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    m = int(row["M"])
                    k = int(row["K"])
                    n = int(row["N"])
                    dtype_str = row["dtype"]

                    if dtype_str in dtype_map:
                        dtype = dtype_map[dtype_str]
                        # Store as (m, k, n, dtype1, dtype2) with same dtype for both
                        shapes.append((m, k, n, dtype, dtype))
                    else:
                        print(
                            f"Warning: Unknown dtype '{dtype_str}' in {csv_file}, skipping row"
                        )

            print(f"Loaded {len(shapes)} shapes from {csv_file}")
        except Exception as e:
            print(f"Error loading shapes from {csv_file}: {e}")

        return shapes

    def setup_shape_collections(self, csv_files: list[str]) -> None:
        """Setup additional shape collections from CSV files and built-in collection."""
        self.additional_shape_collections = []

        # Load shapes from provided CSV files first
        for csv_file in csv_files:
            shapes = self.load_shapes_from_csv(csv_file)
            if shapes:
                self.additional_shape_collections.append(shapes)

        self.additional_shape_collections.append(collect_known_mm_shapes())
        self.shape_generator = self.generate_mm_shapes()

    def generate_mm_shapes(self) -> Generator[tuple[int, int, int, Any], None, None]:
        """Generator that yields (m, k, n, dtype) tuples for matrix multiplication.

        First exhausts all shapes from additional_shape_collections, then generates random shapes.
        Only yields unaligned shapes since external CSV shapes may not be pre-filtered.
        """
        # Phase 1: Use all shapes from additional shape collections
        for collection in self.additional_shape_collections:
            for m, k, n, dtype1, _ in collection:
                # Filter for unaligned shapes only (external CSVs may not be pre-filtered)
                align_size = get_alignment_size_dtype(dtype1)
                if not all(self.is_aligned(dim, align_size) for dim in [m, k, n]):
                    # Check if it fits in memory
                    if fits_in_memory(dtype1, m, k, n):
                        yield (m, k, n, dtype1)

        # Phase 2: Generate infinite random shapes

        while True:
            # Generate random dtype
            dtype_choices = [torch.float16, torch.bfloat16, torch.float32]
            dtype = random.choices(dtype_choices)[0]

            # Generate random shape for this dtype
            uniform = random.choices([True, False])[0]
            align_size = get_alignment_size_dtype(dtype)

            # Keep trying until we get a valid unaligned shape that fits in memory
            while True:
                if uniform:
                    m = random.randint(1, 65536)
                    k = random.randint(1, 65536)
                    n = random.randint(1, 65536)
                else:
                    m = self.get_random_dim()
                    k = self.get_random_dim()
                    n = self.get_random_dim()

                # Skip if all dimensions are aligned (we need unaligned for padding to be relevant)
                if all(self.is_aligned(dim, align_size) for dim in [m, k, n]):
                    continue

                # Check if it fits in memory
                if fits_in_memory(dtype, m, k, n):
                    yield (m, k, n, dtype)
                    break

    def create_input(self) -> tuple[Any, ...]:
        # Get the next shape from the generator
        m, k, n, dtype = next(self.shape_generator)
        set_precision(dtype)

        (transpose_left, transpose_right) = transpose_tensors()
        prepadded_left = self.prepadded()
        prepadded_right = self.prepadded()
        return (
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype,
            prepadded_left,
            prepadded_right,
        )

    def run_benchmark(
        self,
        m: int,
        k: int,
        n: int,
        transpose_left: bool,
        transpose_right: bool,
        dtype: Any,
        prepadded_left: bool,
        prepadded_right: bool,
    ) -> None:
        a, b = get_mm_tensors(
            m,
            k,
            n,
            transpose_left,
            transpose_right,
            dtype_left=dtype,
            dtype_right=dtype,
        )

        print("Benchmarking the following input:")
        print(f"m={m} k={k} n={n} dtype={dtype}")
        print(f"transpose_left={transpose_left} transpose_right={transpose_right}")
        print(f"prepadded_left={prepadded_left} prepadded_right={prepadded_right}")

        with fresh_cache():

            def mm(a: Any, b: Any) -> Any:
                return torch.mm(a, b)

            def mm_mat1_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a + 1, b)

            def mm_mat2_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a, b + 1)

            def mm_mat1_mat2_prepadded(a: Any, b: Any) -> Any:
                return torch.mm(a + 1, b + 1)

            if prepadded_left and prepadded_right:
                cf = torch.compile(mm_mat1_mat2_prepadded)
            elif prepadded_left:
                cf = torch.compile(mm_mat1_prepadded)
            elif prepadded_right:
                cf = torch.compile(mm_mat2_prepadded)
            else:
                cf = torch.compile(mm)
            cf(a, b)
            torch.compiler.reset()

    def get_random_dim(
        self, min_power2: int = 1, max_power2: int = 16, p_unaligned: float = 0.25
    ) -> int:
        aligned = random.choices([True, False], [1 - p_unaligned, p_unaligned])[0]
        if aligned:
            return 2 ** random.randint(min_power2, max_power2)  # type: ignore[no-any-return]
        else:
            # choose a random number between 2^i and 2^(i+1)
            return get_random_between_pow2(min_power2, max_power2)  # type: ignore[no-any-return]

    def is_aligned(self, dim: int, align_size: int) -> bool:
        return dim % align_size == 0

    def prepadded(self, p_prepadded: float = 0.2) -> bool:
        # p_prepadded: probability that a tensor is "prepadded", i.e. pad_mm excludes time it takes to pad from benchmarking
        return random.choices([True, False], [p_prepadded, 1 - p_prepadded])[0]

    def run(self) -> None:
        """Override run to setup shape collections before running."""
        import time

        from tqdm import tqdm

        torch.set_default_device("cuda")
        args = self.parser.parse_args()

        # Setup shape collections based on CLI arguments
        self.setup_shape_collections(args.additional_shape_csv)

        # Set up torch configuration (copied from parent run method)
        if args.use_heuristic:
            torch._inductor.config.autoheuristic_use = self.name
            torch._inductor.config.autoheuristic_collect = ""
        else:
            torch._inductor.config.autoheuristic_use = ""
            torch._inductor.config.autoheuristic_collect = self.name
        torch._inductor.config.autoheuristic_log_path = args.o
        if args.device is not None:
            torch.cuda.set_device(args.device)
        random.seed(time.time())

        # Run the main benchmarking loop
        for _ in tqdm(range(args.num_samples)):
            input = self.create_input()
            for _ in range(args.num_reps):
                self.run_benchmark(*input)


if __name__ == "__main__":
    runner = BenchmarkRunnerPadMM()
    runner.run()
