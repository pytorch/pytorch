#!/usr/bin/env python3
"""
Multi-process fuzzer that uses worker processes to execute fuzzer.py with different seeds.
"""

import argparse
import multiprocessing as mp
import subprocess
import sys
import time
from typing import Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def run_fuzzer_with_seed(seed: int) -> Tuple[int, bool, str, float]:
    """
    Run fuzzer.py with a specific seed.

    Args:
        seed: The seed value to pass to fuzzer.py

    Returns:
        Tuple of (seed, success, output, duration)
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

        return seed, success, output, duration

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        return seed, False, f"Process timed out after 300 seconds", duration

    except Exception as e:
        duration = time.time() - start_time
        return seed, False, f"Exception occurred: {str(e)}", duration


def run_multi_process_fuzzer(
    num_processes: int = 2,
    seed_range: Tuple[int, int] = (1, 10),
    verbose: bool = False
) -> None:
    """
    Run the multi-process fuzzer.

    Args:
        num_processes: Number of worker processes to use
        seed_range: Tuple of (start_seed, end_seed) inclusive
        verbose: Whether to print detailed output
    """
    start_seed, end_seed = seed_range
    seeds = list(range(start_seed, end_seed + 1))

    print(f"🚀 Starting multi-process fuzzer with {num_processes} processes")
    print(f"📊 Processing seeds {start_seed} to {end_seed} ({len(seeds)} total)")
    print(f"🔧 Command template: python fuzzer.py --seed {{seed}}")
    print("=" * 60)

    start_time = time.time()
    results = []
    successful_count = 0
    failed_count = 0

    # Use multiprocessing Pool to distribute work
    with mp.Pool(processes=num_processes) as pool:
        # Submit all seeds to the process pool
        future_results = []
        for seed in seeds:
            future = pool.apply_async(run_fuzzer_with_seed, (seed,))
            future_results.append(future)

        # Set up progress bar
        if HAS_TQDM:
            pbar = tqdm(total=len(seeds), desc="Processing seeds",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] ✅{postfix} ❌{postfix}")
            pbar.set_postfix_str(f"{successful_count}/{failed_count}")
        else:
            print("Progress: (install tqdm for better progress bar)")
            pbar = None

        # Collect results as they complete
        for i, future in enumerate(future_results):
            try:
                seed, success, output, duration = future.get()
                results.append((seed, success, output, duration))

                if success:
                    successful_count += 1
                else:
                    failed_count += 1

                # Update progress bar
                if HAS_TQDM and pbar:
                    pbar.set_postfix_str(f"{successful_count}/{failed_count}")
                    pbar.update(1)
                else:
                    print(f"Completed {i+1}/{len(seeds)} - Seed {seed}: {'✅' if success else '❌'}")

                # Only show detailed output for failures (unless verbose)
                if not success:
                    if HAS_TQDM and pbar:
                        pbar.write(f"❌ FAILURE - Seed {seed} (duration: {duration:.2f}s):")
                        for line in output.split('\n'):
                            if line.strip():
                                pbar.write(f"   {line}")
                        pbar.write("")  # Empty line
                    else:
                        print(f"❌ FAILURE - Seed {seed} (duration: {duration:.2f}s):")
                        for line in output.split('\n'):
                            if line.strip():
                                print(f"   {line}")
                        print()
                elif verbose:
                    if HAS_TQDM and pbar:
                        pbar.write(f"✅ SUCCESS - Seed {seed} (duration: {duration:.2f}s)")
                        if output.strip():
                            for line in output.split('\n'):
                                if line.strip():
                                    pbar.write(f"   {line}")
                            pbar.write("")
                    else:
                        print(f"✅ SUCCESS - Seed {seed} (duration: {duration:.2f}s)")
                        if output.strip():
                            for line in output.split('\n'):
                                if line.strip():
                                    print(f"   {line}")
                            print()

            except Exception as e:
                failed_count += 1
                if HAS_TQDM and pbar:
                    pbar.set_postfix_str(f"{successful_count}/{failed_count}")
                    pbar.update(1)
                    pbar.write(f"❌ POOL ERROR - Seed {seeds[i]}: {str(e)}")
                else:
                    print(f"Completed {i+1}/{len(seeds)} - Seed {seeds[i]}: ❌ POOL ERROR")
                    print(f"❌ POOL ERROR - Seed {seeds[i]}: {str(e)}")
                results.append((seeds[i], False, f"Pool error: {str(e)}", 0.0))

        # Close progress bar
        if HAS_TQDM and pbar:
            pbar.close()

    total_time = time.time() - start_time

    # Print summary
    print("=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"✅ Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"❌ Failed:     {len(failed)}/{len(results)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"⚡ Avg time per seed: {sum(r[3] for r in results)/len(results):.2f}s")

    if failed:
        print(f"\n❌ Failed seeds: {[r[0] for r in failed]}")

    if successful:
        print(f"✅ Successful seeds: {[r[0] for r in successful]}")
        avg_success_time = sum(r[3] for r in successful) / len(successful)
        print(f"⚡ Avg time for successful runs: {avg_success_time:.2f}s")


def main():
    """Main entry point for the multi-process fuzzer."""
    parser = argparse.ArgumentParser(
        description="Multi-process fuzzer for fuzzer.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: auto-detected processes, seeds 1-10
  python multi_process_fuzzer.py

  # Use 4 processes with seeds 1-20
  python multi_process_fuzzer.py --processes 4 --start 1 --end 20

  # Use 1 process with seeds 5-8, verbose output
  python multi_process_fuzzer.py --processes 1 --start 5 --end 8 --verbose

  # Auto-detect processes, custom seed range
  python multi_process_fuzzer.py --start 100 --end 200
        """
    )

    # Auto-detect optimal number of processes
    cpu_count = mp.cpu_count()
    # Use 75% of available CPUs, minimum 1, maximum 8 (for reasonable resource usage)
    default_processes = max(1, min(8, int(cpu_count * 0.75)))

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
        "--end",
        type=int,
        default=100,
        help="Ending seed value (inclusive, default: 10)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for all runs (not just failures)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.processes < 1:
        print("❌ Error: Number of processes must be at least 1")
        sys.exit(1)

    if args.start > args.end:
        print("❌ Error: Start seed must be <= end seed")
        sys.exit(1)

    # Check if fuzzer.py exists
    import os
    if not os.path.exists("fuzzer.py"):
        print("❌ Error: fuzzer.py not found in current directory")
        print("Make sure you're running this script from the correct directory")
        sys.exit(1)

    try:
        run_multi_process_fuzzer(
            num_processes=args.processes,
            seed_range=(args.start, args.end),
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user (Ctrl+C)")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
