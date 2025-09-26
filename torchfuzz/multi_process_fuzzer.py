#!/usr/bin/env python3
"""
Multi-process fuzzer that uses worker processes to execute fuzzer.py with different seeds.
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
import re
from typing import Tuple, List, Dict

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def persist_print(msg):
    try:
        if HAS_TQDM:
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
IGNORE_PATTERNS: List[re.Pattern] = [
    re.compile(r"Dynamo failed to run FX node with fake tensors: call_method fill_diagonal_"),
    re.compile(r"zuf\d+ is not defined"),
    re.compile(r"‘zuf\d+’ was not declared in this scope"),
    re.compile(r"out_eager_sum: 0\.0"),
    re.compile(r"error: operand #\d+ does not dominate this use"),
    re.compile(r"TypeError: unsupported operand type\(s\) for divmod\(\): 'SymInt' and 'int'"),
    re.compile(r"torch\._inductor\.exc\.InductorError: CantSplit:"),
    re.compile(r"Error in op: torch\.ops\.aten\.convolution_backward\.default"),
    re.compile(r"RuntimeError: PassManager::run failed"),
    re.compile(r"AttributeError: 'Infinity' object has no attribute '_mpf_'"),
    re.compile(r"AssertionError: Node full_1 was invalid, but is output"),
    re.compile(r"Error in op: torch\.ops\.aten\.convolution\.default"),
    # Add more patterns here as needed, e.g.:
    # re.compile(r"Some other error message"),
]

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

def run_fuzzer_with_seed(seed: int) -> Tuple[int, bool, str, float, int]:
    """
    Run fuzzer.py with a specific seed.

    Args:
        seed: The seed value to pass to fuzzer.py

    Returns:
        Tuple of (seed, success, output, duration, ignored_pattern_idx)
        ignored_pattern_idx: -1 if not ignored, otherwise index of IGNORE_PATTERNS
    """
    start_time = time.time()

    try:
        # Run fuzzer.py with the specified seed
        cmd = [sys.executable, "fuzzer.py", "--seed", str(seed)]

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

        # Check if output should be ignored and which pattern matched
        ignored_pattern_idx = is_ignored_output(output)
        if ignored_pattern_idx != -1:
            # Mark as ignored (could also return a special flag if needed)
            output = "[IGNORED] " + output

        return seed, success, output, duration, ignored_pattern_idx

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return seed, False, f"Process timed out after 300 seconds", duration, -1

    except Exception as e:
        duration = time.time() - start_time
        return seed, False, f"Exception occurred: {str(e)}", duration, -1


def run_multi_process_fuzzer(
    num_processes: int = 2,
    seed_start: int = 1,
    seed_count: int = 10,
    verbose: bool = False
) -> None:
    """
    Run the multi-process fuzzer.

    Args:
        num_processes: Number of worker processes to use
        seed_start: Starting seed value (inclusive)
        seed_count: Number of seeds to run
        verbose: Whether to print detailed output
    """
    seeds = list(range(seed_start, seed_start + seed_count))

    persist_print(f"🚀 Starting multi-process fuzzer with {num_processes} processes")
    persist_print(f"📊 Processing seeds {seed_start} to {seed_start + seed_count - 1} ({len(seeds)} total)")
    persist_print(f"🔧 Command template: python fuzzer.py --seed {{seed}}")
    persist_print("=" * 60)

    start_time = time.time()
    results = []
    successful_count = 0
    failed_count = 0
    ignored_count = 0
    ignored_seeds = []
    ignored_pattern_counts: Dict[int, int] = {i: 0 for i in range(len(IGNORE_PATTERNS))}

    try:
        # Use multiprocessing Pool to distribute work
        with mp.Pool(processes=num_processes) as pool:
            # Submit all seeds to the process pool
            future_results = []
            for seed in seeds:
                future = pool.apply_async(run_fuzzer_with_seed, (seed,))
                future_results.append(future)

            # Set up progress bar
            if HAS_TQDM:
                pbar = tqdm(
                    total=len(seeds),
                    desc="Processing seeds",
                    file=sys.stdout,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅/❌/❓={postfix}",
                    dynamic_ncols=True
                )
                pbar.set_postfix_str(f"{successful_count}/{failed_count}/{ignored_count} | throughput: 0.00 seeds/hr")
            else:
                persist_print("Progress: (install tqdm for better progress bar)")
                pbar = None

            # Collect results as they complete
            for i, future in enumerate(future_results):
                try:
                    seed, success, output, duration, ignored_pattern_idx = future.get()
                    results.append((seed, success, output, duration, ignored_pattern_idx))

                    if ignored_pattern_idx != -1:
                        ignored_seeds.append(seed)
                        ignored_pattern_counts[ignored_pattern_idx] += 1
                        ignored_count += 1

                    # Only increment failed_count if not ignored
                    if success:
                        successful_count += 1
                    elif ignored_pattern_idx == -1:
                        failed_count += 1

                    elapsed = time.time() - start_time
                    throughput = (i + 1) / (elapsed / 3600)

                    # Update progress bar
                    if HAS_TQDM and pbar:
                        pbar.set_postfix_str(f"{successful_count}/{failed_count}/{ignored_count} | throughput: {throughput:.2f} seeds/hr")
                        # tqdm automatically shows ETA (estimated time remaining) in the bar_format above
                        pbar.update(1)
                    else:
                        persist_print(f"Completed {i+1}/{len(seeds)} - Seed {seed}: {'✅' if success else '❌'}{' (IGNORED)' if ignored_pattern_idx != -1 else ''}")

                    # Only show detailed output for failures (unless verbose)
                    if not success and ignored_pattern_idx == -1:
                        if HAS_TQDM and pbar:
                            pbar.write(f"❌ FAILURE - Seed {seed} (duration: {duration:.2f}s):")
                            for line in output.split('\n'):
                                if line.strip():
                                    pbar.write(f"   {line}")
                            pbar.write("")  # Empty line
                        else:
                            persist_print(f"❌ FAILURE - Seed {seed} (duration: {duration:.2f}s):")
                            for line in output.split('\n'):
                                if line.strip():
                                    persist_print(f"   {line}")
                            persist_print("")
                    elif not success and ignored_pattern_idx != -1:
                        # Optionally, print ignored failures if desired
                        if verbose:
                            if HAS_TQDM and pbar:
                                pbar.write(f"🚫 IGNORED - Seed {seed} (duration: {duration:.2f}s):")
                                for line in output.split('\n'):
                                    if line.strip():
                                        pbar.write(f"   {line}")
                                pbar.write("")
                            else:
                                persist_print(f"🚫 IGNORED - Seed {seed} (duration: {duration:.2f}s):")
                                for line in output.split('\n'):
                                    if line.strip():
                                        persist_print(f"   {line}")
                                persist_print("")
                    elif verbose:
                        if HAS_TQDM and pbar:
                            pbar.write(f"✅ SUCCESS - Seed {seed} (duration: {duration:.2f}s){' [IGNORED]' if ignored_pattern_idx != -1 else ''}")
                            if output.strip():
                                for line in output.split('\n'):
                                    if line.strip():
                                        pbar.write(f"   {line}")
                                pbar.write("")
                        else:
                            persist_print(f"✅ SUCCESS - Seed {seed} (duration: {duration:.2f}s){' [IGNORED]' if ignored_pattern_idx != -1 else ''}")
                            if output.strip():
                                for line in output.split('\n'):
                                    if line.strip():
                                        persist_print(f"   {line}")
                                persist_print("")

                except Exception as e:
                    failed_count += 1
                    if HAS_TQDM and pbar:
                        pbar.set_postfix_str(f"{successful_count}/{failed_count}")
                        pbar.update(1)
                        pbar.write(f"❌ POOL ERROR - Seed {seeds[i]}: {str(e)}")
                    else:
                        persist_print(f"Completed {i+1}/{len(seeds)} - Seed {seeds[i]}: ❌ POOL ERROR")
                        persist_print(f"❌ POOL ERROR - Seed {seeds[i]}: {str(e)}")
                    results.append((seeds[i], False, f"Pool error: {str(e)}", 0.0, -1))

            # Close progress bar
            if HAS_TQDM and pbar:
                pbar.close()
    except KeyboardInterrupt as e:
        persist_print("\n🛑 Interrupted by user (Ctrl+C)")
        # Print summary up to this point
        total_time = time.time() - start_time
        persist_print("=" * 60)
        persist_print("📈 SUMMARY (partial, interrupted)")
        persist_print("=" * 60)

        successful = [r for r in results if r[1]]
        # Only count as failed if not ignored
        failed = [r for r in results if not r[1] and r[4] == -1]
        ignored = [r for r in results if r[4] != -1]

        persist_print(f"✅ Successful: {len(successful)}/{len(results)} ({(len(successful)/len(results)*100 if results else 0):.1f}%)")
        persist_print(f"❌ Failed:     {len(failed)}/{len(results)} ({(len(failed)/len(results)*100 if results else 0):.1f}%)")
        persist_print(f"⏱️  Total time: {total_time:.2f}s")
        if results:
            persist_print(f"⚡ Throughput: {(len(results) / (total_time / 3600)):.2f} seeds/hr" if total_time > 0 else "⚡ Throughput: N/A")
        if failed:
            persist_print(f"\n❌ Failed seeds: {[r[0] for r in failed]}")
        if ignored:
            persist_print(f"\n🚫 Ignored seeds: {[r[0] for r in ignored]}")
            # Print ignore pattern stats
            persist_print("\n🚫 Ignored pattern statistics:")
            total_ignored = len(ignored)
            for idx, pattern in enumerate(IGNORE_PATTERNS):
                count = ignored_pattern_counts[idx]
                percent = (count / total_ignored * 100) if total_ignored else 0
                persist_print(f"  Pattern {idx}: {pattern.pattern!r} - {count} ({percent:.1f}%)")

        sys.exit(130)

    total_time = time.time() - start_time

    # Print summary
    persist_print("=" * 60)
    persist_print("📈 SUMMARY")
    persist_print("=" * 60)

    successful = [r for r in results if r[1]]
    # Only count as failed if not ignored
    failed = [r for r in results if not r[1] and r[4] == -1]
    ignored = [r for r in results if r[4] != -1]

    persist_print(f"✅ Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    persist_print(f"❌ Failed:     {len(failed)}/{len(results)} ({len(failed)/len(results)*100:.1f}%)")
    persist_print(f"⏱️  Total time: {total_time:.2f}s")
    persist_print(f"⚡ Throughput: {(len(results) / (total_time / 3600)):.2f} seeds/hr" if total_time > 0 else "⚡ Throughput: N/A")

    if failed:
        persist_print(f"\n❌ Failed seeds: {[r[0] for r in failed]}")

    if ignored:
        persist_print(f"\n🚫 Ignored seeds: {[r[0] for r in ignored]}")
        # Print ignore pattern stats
        persist_print("\n🚫 Ignored pattern statistics:")
        total_ignored = len(ignored)
        for idx, pattern in enumerate(IGNORE_PATTERNS):
            count = ignored_pattern_counts[idx]
            percent = (count / total_ignored * 100) if total_ignored else 0
            persist_print(f"  Pattern {idx}: {pattern.pattern!r} - {count} ({percent:.1f}%)")



def main():
    """Main entry point for the multi-process fuzzer."""
    parser = argparse.ArgumentParser(
        description="Multi-process fuzzer for fuzzer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    cpu_count = mp.cpu_count()
    default_processes = max(1, min(16, int(cpu_count * 0.75)))

    parser.add_argument(
        "--processes", "-p",
        type=int,
        default=default_processes,
        help=f"Number of worker processes to use (default: {default_processes}, auto-detected from {cpu_count} CPUs)"
    )

    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting seed value (inclusive, default: 1)"
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of seeds to run (default: 10)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for all runs (not just failures)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.processes < 1:
        persist_print("❌ Error: Number of processes must be at least 1")
        sys.exit(1)

    if args.count < 1:
        persist_print("❌ Error: Count must be at least 1")
        sys.exit(1)

    import os
    if not os.path.exists("fuzzer.py"):
        persist_print("❌ Error: fuzzer.py not found in current directory")
        sys.exit(1)

    try:
        run_multi_process_fuzzer(
            num_processes=args.processes,
            seed_start=args.start,
            seed_count=args.count,
            verbose=args.verbose
        )
    except Exception as e:
        persist_print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
