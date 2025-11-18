#!/usr/bin/env python3
"""
Multi-process fuzzer library that uses worker processes to execute fuzzer.py with different seeds.
"""

import multiprocessing as mp
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

    # Create a mock tqdm class for type safety
    class MockTqdm:
        @staticmethod
        def write(msg, file=None):
            print(msg, file=file, flush=True)

    tqdm = MockTqdm()


def persist_print(msg):
    """Print messages that persist with tqdm progress bars."""
    try:
        if HAS_TQDM and hasattr(tqdm, "write"):
            # Keep prints on the same stream as the bar
            tqdm.write(msg, file=sys.stderr)
        else:
            print(msg, file=sys.stderr, flush=True)
    except BrokenPipeError:
        import os

        os.makedirs("/tmp/torchfuzz", exist_ok=True)
        with open("/tmp/torchfuzz/crash.log", "a") as f:
            f.write(f"BrokenPipeError: {msg}\n")


# List of regex patterns for ignore bucket
IGNORE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"torch\._inductor\.exc\.InductorError: AssertionError: -1"
    ),  # https://github.com/pytorch/pytorch/issues/167937
    # Add more patterns here as needed, e.g.:
    # re.compile(r"Some other error message"),
]


@dataclass
class FuzzerResult:
    seed: int
    success: bool
    output: str
    duration: float
    ignored_pattern_idx: int
    operation_stats: dict[str, int]  # New field for operation statistics


def is_ignored_output(output: str) -> int:
    """
    Check if the output matches any ignore pattern.

    Args:
        output: The combined stdout/stderr string.

    Returns:
        Index of the matched ignore pattern, or -1 if none matched.
    """
    for idx, pattern in enumerate(IGNORE_PATTERNS):
        if pattern.search(output):
            return idx
    return -1


def run_fuzzer_with_seed(
    seed: int,
    template: str = "default",
    supported_ops: Optional[str] = None,
) -> FuzzerResult:
    """
    Run fuzzer.py with a specific seed.

    Args:
        seed: The seed value to pass to fuzzer.py
        template: The template to use for code generation
        supported_ops: Comma-separated ops string with optional weights

    Returns:
        FuzzerResult dataclass instance
    """
    start_time = time.time()

    try:
        # Run fuzzer.py with the specified seed and template
        cmd = [
            sys.executable,
            "fuzzer.py",
            "--single",
            "--seed",
            str(seed),
            "--template",
            template,
        ]

        # Append supported ops if provided
        if supported_ops:
            cmd.extend(["--supported-ops", supported_ops])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per seed
        )

        duration = time.time() - start_time
        success = result.returncode == 0

        # Combine stdout and stderr for output
        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}\n"
        if result.stderr:
            output += f"STDERR:\n{result.stderr}\n"
        output += f"Return code: {result.returncode}"

        # Parse operation statistics from the output
        operation_stats = {}
        if result.stdout:
            lines = result.stdout.split("\n")
            in_stats_section = False
            for line in lines:
                if line.strip() == "OPERATION_STATS:":
                    in_stats_section = True
                    continue
                elif in_stats_section:
                    if line.startswith("  ") and ":" in line:
                        # Parse line like "  torch.add: 3"
                        op_line = line.strip()
                        if ": " in op_line:
                            op_name, count_str = op_line.split(": ", 1)
                            try:
                                count = int(count_str)
                                operation_stats[op_name] = count
                            except ValueError:
                                pass  # Skip malformed lines
                    else:
                        # End of stats section
                        in_stats_section = False

        # Check if output should be ignored and which pattern matched
        ignored_pattern_idx = is_ignored_output(output)
        if ignored_pattern_idx != -1:
            # Mark as ignored (could also return a special flag if needed)
            output = "[IGNORED] " + output

        return FuzzerResult(
            seed, success, output, duration, ignored_pattern_idx, operation_stats
        )

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return FuzzerResult(
            seed, False, "Process timed out after 300 seconds", duration, -1, {}
        )

    except Exception as e:
        duration = time.time() - start_time
        return FuzzerResult(
            seed, False, f"Exception occurred: {str(e)}", duration, -1, {}
        )


def print_output_lines(output: str, write_func):
    """Helper to print non-empty lines of output using the provided write_func."""
    for line in output.split("\n"):
        if line.strip():
            write_func(f"   {line}")
    if hasattr(write_func, "__self__") and hasattr(write_func.__self__, "write"):
        # For tqdm.write, add an empty line for separation
        write_func("")


def handle_result_output(
    *,
    label: str,
    seed: int,
    duration: float,
    output: str,
    ignored: bool,
    verbose: bool,
    write_func,
):
    """Unified handler for result output, reducing code repetition."""
    ignored_text = " [IGNORED]" if ignored else ""
    write_func(f"{label} - Seed {seed} (duration: {duration:.2f}s){ignored_text}")
    if output.strip() or label.startswith("❌") or verbose:
        print_output_lines(output, write_func)


def run_multi_process_fuzzer(
    num_processes: Optional[int] = None,
    seed_start: int = 0,
    seed_count: int = 100,
    verbose: bool = False,
    template: str = "default",
    supported_ops: Optional[str] = None,
) -> None:
    """
    Run the multi-process fuzzer.

    Args:
        num_processes: Number of worker processes to use
        seed_start: Starting seed value (inclusive)
        seed_count: Number of seeds to run
        verbose: Whether to print detailed output
        template: The template to use for code generation
        supported_ops: Comma-separated ops string with optional weights
    """
    seeds = list(range(seed_start, seed_start + seed_count))

    persist_print(f"🚀 Starting multi-process fuzzer with {num_processes} processes")
    persist_print(
        f"📊 Processing seeds {seed_start} to {seed_start + seed_count - 1} ({len(seeds)} total)"
    )
    persist_print(
        f"🔧 Command template: python fuzzer.py --seed {{seed}} --template {template}"
    )
    persist_print("=" * 60)

    start_time = time.time()
    results: list[FuzzerResult] = []
    successful_count = 0
    failed_count = 0
    ignored_count = 0
    ignored_seeds = []
    ignored_pattern_counts: dict[int, int] = dict.fromkeys(
        range(len(IGNORE_PATTERNS)), 0
    )

    try:
        # Use multiprocessing Pool to distribute work
        with mp.Pool(processes=num_processes) as pool:
            # Submit all seeds to the process pool
            future_results = []
            for seed in seeds:
                future = pool.apply_async(
                    run_fuzzer_with_seed, (seed, template, supported_ops)
                )
                future_results.append(future)

            # Set up progress bar
            if HAS_TQDM:
                from tqdm import tqdm  # Import the real tqdm here

                pbar = tqdm(
                    total=len(seeds),
                    desc="Processing seeds",
                    file=sys.stdout,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅/❌/❓={postfix}",
                    dynamic_ncols=True,
                )
                pbar.set_postfix_str(
                    f"{successful_count}/{failed_count}/{ignored_count} | throughput: 0.00 seeds/hr"
                )

                def write_func(msg):
                    # pyrefly: ignore [missing-attribute]
                    pbar.write(msg)
            else:
                persist_print("Progress: (install tqdm for better progress bar)")
                pbar = None
                write_func = persist_print

            # Collect results as they complete
            for i, future in enumerate(future_results):
                try:
                    result: FuzzerResult = future.get()
                    results.append(result)

                    if result.ignored_pattern_idx != -1:
                        ignored_seeds.append(result.seed)
                        ignored_pattern_counts[result.ignored_pattern_idx] += 1
                        ignored_count += 1

                    # Only increment failed_count if not ignored
                    if result.success:
                        successful_count += 1
                    elif result.ignored_pattern_idx == -1:
                        failed_count += 1

                    elapsed = time.time() - start_time
                    throughput = (i + 1) / (elapsed / 3600)

                    # Update progress bar
                    if HAS_TQDM and pbar:
                        pbar.set_postfix_str(
                            f"{successful_count}/{failed_count}/{ignored_count} | throughput: {throughput:.2f} seeds/hr"
                        )
                        pbar.update(1)
                    else:
                        status_emoji = "✅" if result.success else "❌"
                        ignored_text = (
                            " (IGNORED)" if result.ignored_pattern_idx != -1 else ""
                        )
                        persist_print(
                            f"Completed {i + 1}/{len(seeds)} - Seed {result.seed}: {status_emoji}{ignored_text}"
                        )

                    # Unified output handling
                    if not result.success and result.ignored_pattern_idx == -1:
                        handle_result_output(
                            label="❌ FAILURE",
                            seed=result.seed,
                            duration=result.duration,
                            output=result.output,
                            ignored=False,
                            verbose=verbose,
                            write_func=write_func,
                        )
                    elif not result.success and result.ignored_pattern_idx != -1:
                        if verbose:
                            handle_result_output(
                                label="🚫 IGNORED",
                                seed=result.seed,
                                duration=result.duration,
                                output=result.output,
                                ignored=True,
                                verbose=verbose,
                                write_func=write_func,
                            )
                    elif verbose:
                        handle_result_output(
                            label="✅ SUCCESS",
                            seed=result.seed,
                            duration=result.duration,
                            output=result.output,
                            ignored=(result.ignored_pattern_idx != -1),
                            verbose=verbose,
                            write_func=write_func,
                        )

                except Exception as e:
                    failed_count += 1
                    if HAS_TQDM and pbar:
                        pbar.set_postfix_str(f"{successful_count}/{failed_count}")
                        pbar.update(1)
                        pbar.write(f"❌ POOL ERROR - Seed {seeds[i]}: {str(e)}")
                    else:
                        persist_print(
                            f"Completed {i + 1}/{len(seeds)} - Seed {seeds[i]}: ❌ POOL ERROR"
                        )
                        persist_print(f"❌ POOL ERROR - Seed {seeds[i]}: {str(e)}")
                    results.append(
                        FuzzerResult(
                            seeds[i], False, f"Pool error: {str(e)}", 0.0, -1, {}
                        )
                    )

            # Close progress bar
            if HAS_TQDM and pbar:
                pbar.close()
    except KeyboardInterrupt:
        persist_print("\n🛑 Interrupted by user (Ctrl+C)")
        # Print summary up to this point
        total_time = time.time() - start_time
        persist_print("=" * 60)
        persist_print("📈 SUMMARY (partial, interrupted)")
        persist_print("=" * 60)

        successful = [res for res in results if res.success]
        # Only count as failed if not ignored
        failed = [
            res for res in results if not res.success and res.ignored_pattern_idx == -1
        ]
        ignored = [res for res in results if res.ignored_pattern_idx != -1]

        persist_print(
            f"✅ Successful: {len(successful)}/{len(results)} ({(len(successful) / len(results) * 100 if results else 0):.1f}%)"
        )
        persist_print(
            f"❌ Failed:     {len(failed)}/{len(results)} ({(len(failed) / len(results) * 100 if results else 0):.1f}%)"
        )
        persist_print(f"⏱️  Total time: {total_time:.2f}s")
        if results:
            persist_print(
                f"⚡ Throughput: {(len(results) / (total_time / 3600)):.2f} seeds/hr"
                if total_time > 0
                else "⚡ Throughput: N/A"
            )
        if failed:
            persist_print(f"\n❌ Failed seeds: {[res.seed for res in failed]}")
        if successful:
            persist_print(f"✅ Successful seeds: {[res.seed for res in successful]}")
            avg_success_time = sum(res.duration for res in successful) / len(successful)
            persist_print(f"⚡ Avg time for successful runs: {avg_success_time:.2f}s")
        if ignored:
            persist_print(f"\n🚫 Ignored seeds: {[res.seed for res in ignored]}")
            # Print ignore pattern stats
            persist_print("\n🚫 Ignored pattern statistics:")
            total_ignored = len(ignored)
            for idx, pattern in enumerate(IGNORE_PATTERNS):
                count = ignored_pattern_counts[idx]
                percent = (count / total_ignored * 100) if total_ignored else 0
                persist_print(
                    f"  Pattern {idx}: {pattern.pattern!r} - {count} ({percent:.1f}%)"
                )

        # Aggregate and print operation distribution
        _print_operation_distribution(results)

        sys.exit(130)

    total_time = time.time() - start_time

    # Print summary
    persist_print("=" * 60)
    persist_print("📈 SUMMARY")
    persist_print("=" * 60)

    successful = [res for res in results if res.success]
    # Only count as failed if not ignored
    failed = [
        res for res in results if not res.success and res.ignored_pattern_idx == -1
    ]
    ignored = [res for res in results if res.ignored_pattern_idx != -1]

    persist_print(
        f"✅ Successful: {len(successful)}/{len(results)} ({len(successful) / len(results) * 100:.1f}%)"
    )
    persist_print(
        f"❌ Failed:     {len(failed)}/{len(results)} ({len(failed) / len(results) * 100:.1f}%)"
    )
    persist_print(f"⏱️  Total time: {total_time:.2f}s")
    persist_print(
        f"⚡ Throughput: {(len(results) / (total_time / 3600)):.2f} seeds/hr"
        if total_time > 0
        else "⚡ Throughput: N/A"
    )

    if failed:
        persist_print(f"\n❌ Failed seeds: {[res.seed for res in failed]}")

    if successful:
        persist_print(f"✅ Successful seeds: {[res.seed for res in successful]}")
        avg_success_time = sum(res.duration for res in successful) / len(successful)
        persist_print(f"⚡ Avg time for successful runs: {avg_success_time:.2f}s")

    if ignored:
        persist_print(f"\n🚫 Ignored seeds: {[res.seed for res in ignored]}")
        # Print ignore pattern stats
        persist_print("\n🚫 Ignored pattern statistics:")
        total_ignored = len(ignored)
        for idx, pattern in enumerate(IGNORE_PATTERNS):
            count = ignored_pattern_counts[idx]
            percent = (count / total_ignored * 100) if total_ignored else 0
            persist_print(
                f"  Pattern {idx}: {pattern.pattern!r} - {count} ({percent:.1f}%)"
            )

    # Aggregate and print operation distribution
    _print_operation_distribution(results)


def _print_operation_distribution(results: list[FuzzerResult]) -> None:
    """Helper function to print operation distribution statistics."""
    total_operation_stats = defaultdict(int)
    total_operations = 0

    # Collect operation stats from all successful results
    for result in results:
        if result.success and result.operation_stats:
            for op_name, count in result.operation_stats.items():
                total_operation_stats[op_name] += count
                total_operations += count

    if total_operation_stats:
        persist_print("\n📊 OPERATION DISTRIBUTION")
        persist_print("=" * 60)
        persist_print(f"Total operations executed: {total_operations}")
        persist_print("")

        # Sort operations by count (descending) for better readability
        sorted_ops = sorted(
            total_operation_stats.items(), key=lambda x: x[1], reverse=True
        )

        for op_name, count in sorted_ops:
            percentage = (count / total_operations * 100) if total_operations > 0 else 0
            persist_print(f"  {op_name:<30} {count:>6} times ({percentage:>5.1f}%)")
    else:
        persist_print(
            "\n📊 No operation statistics collected (no successful runs with stats)"
        )


def run_until_failure(
    num_processes: Optional[int] = None,
    verbose: bool = False,
    template: str = "default",
    supported_ops: Optional[str] = None,
) -> None:
    """
    Run the multi-process fuzzer with a random starting seed, iterating until a failure is found.

    Args:
        num_processes: Number of worker processes to use
        verbose: Whether to print detailed output
        template: The template to use for code generation
        supported_ops: Comma-separated ops string with optional weights

    Returns:
        Exits with non-zero code when a failure is found
    """
    import random

    # Pick a random seed to start from
    initial_seed = random.randint(0, 2**31 - 1)

    persist_print(
        f"🎲 Starting continuous fuzzing with random initial seed: {initial_seed}"
    )
    persist_print(f"🚀 Using {num_processes} processes")
    persist_print(
        f"🔧 Command template: python fuzzer.py --seed {{seed}} --template {template}"
    )
    persist_print("🎯 Running until first failure is found...")
    persist_print("=" * 60)

    start_time = time.time()
    current_seed = initial_seed
    total_successful = 0
    total_ignored = 0
    batch_size = 100  # Process seeds in batches of 100

    try:
        while True:
            # Process a batch of seeds
            seeds = list(range(current_seed, current_seed + batch_size))

            with mp.Pool(processes=num_processes) as pool:
                future_results = []
                for seed in seeds:
                    future = pool.apply_async(
                        run_fuzzer_with_seed, (seed, template, supported_ops)
                    )
                    future_results.append((seed, future))

                # Set up progress bar for this batch
                if HAS_TQDM:
                    from tqdm import tqdm

                    pbar = tqdm(
                        total=len(seeds),
                        desc=f"Batch starting at seed {current_seed}",
                        file=sys.stdout,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}] ✅/🚫={postfix}",
                        dynamic_ncols=True,
                    )
                    pbar.set_postfix_str(f"{total_successful}/{total_ignored}")

                    def write_func(msg):
                        # pyrefly: ignore [missing-attribute]
                        pbar.write(msg)
                else:
                    pbar = None

                # Collect results as they complete
                for seed, future in future_results:
                    result: FuzzerResult = future.get()

                    if result.ignored_pattern_idx != -1:
                        total_ignored += 1

                    if result.success:
                        total_successful += 1
                    elif result.ignored_pattern_idx == -1:
                        # Found a failure that is not ignored!
                        if HAS_TQDM and pbar:
                            pbar.close()

                        elapsed = time.time() - start_time
                        persist_print("\n" + "=" * 60)
                        persist_print("🎯 FAILURE FOUND!")
                        persist_print("=" * 60)
                        persist_print(f"❌ Failing seed: {result.seed}")
                        persist_print(
                            f"⏱️  Duration for this seed: {result.duration:.2f}s"
                        )
                        persist_print(f"⏱️  Total time elapsed: {elapsed:.2f}s")
                        persist_print(f"✅ Successful seeds tested: {total_successful}")
                        persist_print(f"🚫 Ignored seeds: {total_ignored}")
                        persist_print(
                            f"📊 Total seeds tested: {total_successful + total_ignored + 1}"
                        )
                        persist_print("\n💥 Failure output:")
                        persist_print("-" * 60)
                        print_output_lines(result.output, persist_print)
                        persist_print("-" * 60)
                        persist_print(
                            f"\n🔄 Reproduce with: python fuzzer.py --seed {result.seed} --template {template}"
                        )

                        # Exit with non-zero code
                        sys.exit(1)

                    # Update progress bar
                    if HAS_TQDM and pbar:
                        pbar.set_postfix_str(f"{total_successful}/{total_ignored}")
                        pbar.update(1)
                    elif verbose:
                        status_emoji = "✅" if result.success else "🚫"
                        persist_print(f"Seed {result.seed}: {status_emoji}")

                # Close progress bar for this batch
                if HAS_TQDM and pbar:
                    pbar.close()

            # Move to next batch
            current_seed += batch_size

    except KeyboardInterrupt:
        persist_print("\n🛑 Interrupted by user (Ctrl+C)")
        elapsed = time.time() - start_time
        persist_print("=" * 60)
        persist_print("📈 SUMMARY (interrupted)")
        persist_print("=" * 60)
        persist_print(f"⏱️  Total time: {elapsed:.2f}s")
        persist_print(f"✅ Successful seeds: {total_successful}")
        persist_print(f"🚫 Ignored seeds: {total_ignored}")
        persist_print(f"📊 Total seeds tested: {total_successful + total_ignored}")
        persist_print(
            f"⚡ Throughput: {((total_successful + total_ignored) / (elapsed / 3600)):.2f} seeds/hr"
        )
        sys.exit(130)
