import copy
import csv
import json
import sys
from dataclasses import dataclass

import torch._logging.scribe as scribe


@dataclass
class ExpectedFileEntry:
    benchmark_name: str
    metric_name: str
    expected_value: int
    noise_margin: float


@dataclass
class ResultFileEntry:
    benchmark_name: str
    metric_name: str
    actual_value: int


def replace_with_zeros(num):
    """
    Keeps the first three digits of an integer and replaces the rest with zeros.

    Args:
        num (int): The number to modify.

    Returns:
        int: The modified number.

    Raises:
        ValueError: If the input is not an integer.
    """
    # Check if input is an integer
    if not isinstance(num, int):
        raise ValueError("Input must be an integer")

    # Calculate the number of digits to remove
    digits_to_remove = len(str(abs(num))) - 4

    # Replace digits with zeros
    if digits_to_remove > 0:
        modified_num = (num // 10**digits_to_remove) * 10**digits_to_remove
    else:
        modified_num = num

    return modified_num


def main():
    # Expected file is the file that have the results that we are comparing against.
    # Expected has the following format:
    # benchmark_name, metric name, expected value, noise margin (as percentage)
    # Example:
    # add_loop_eager,compile_time_instruction_count,283178305, 0.01 (1% noise margin)
    expected_file_path = sys.argv[1]

    # Result file is the file that have the results of the current run. It has the following format:
    # benchmark_name, metric name, expected value, noise margin (as percentage)
    # Example:
    # add_loop_eager,compile_time_instruction_count,283178305
    result_file_path = sys.argv[2]

    # A path where a new expected results file will be written that can be used to replace expected_results.csv
    # in case of failure. In case of no failure the content of this file will match expected_file_path.
    reference_expected_results_path = sys.argv[3]

    # Read expected data file.
    expected_data: dict[str, ExpectedFileEntry] = {}

    with open(expected_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 0:
                continue
            entry = ExpectedFileEntry(
                benchmark_name=row[0].strip(),
                metric_name=row[1].strip(),
                expected_value=int(row[2]),
                noise_margin=float(row[3]),
            )
            key = (entry.benchmark_name, entry.metric_name)
            assert key not in expected_data, f"Duplicate entry for {key}"
            expected_data[key] = entry

    # Read result data file.
    result_data: dict[str, ResultFileEntry] = {}

    with open(result_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            entry = ResultFileEntry(
                benchmark_name=row[0].strip(),
                metric_name=row[1].strip(),
                actual_value=int(row[2]),
            )

            key = (entry.benchmark_name, entry.metric_name)
            assert key not in result_data, f"Duplicate entry for {key}"
            result_data[key] = entry

    fail = False
    new_expected = copy.deepcopy(expected_data)
    for key, entry in expected_data.items():
        if key not in result_data:
            print(f"Missing entry for {key} in result file")
            sys.exit(1)

        low = entry.expected_value - entry.expected_value * entry.noise_margin
        high = entry.expected_value + entry.expected_value * entry.noise_margin
        result = result_data[key].actual_value
        ratio = float(result - entry.expected_value) * 100 / entry.expected_value

        def log(event_name):
            scribe.open_source_signpost(
                subsystem="pr_time_benchmarks",
                name=event_name,
                parameters=json.dumps(
                    {
                        "benchmark_name": entry.benchmark_name,
                        "metric_name": entry.metric_name,
                        "actual_value": result,
                        "expected_value": entry.expected_value,
                        "noise_margin": entry.noise_margin,
                        "change_ratio": ratio,
                    }
                ),
            )

        new_entry = copy.deepcopy(entry)
        # only change if abs(ratio) > entry.noise_margin /3.
        new_entry.expected_value = (
            replace_with_zeros(result)
            if abs(ratio) > entry.noise_margin * 100 / 3
            else entry.expected_value
        )
        new_expected[key] = new_entry

        if result > high:
            fail = True
            print(
                f"REGRESSION: benchmark {key} failed, actual result {result} "
                f"is {ratio:.2f}% higher than expected {entry.expected_value} ±{entry.noise_margin*100:+.2f}% "
                f"if this is an expected regression, please update the expected results.\n"
            )
            print(
                "please update all results that changed significantly, and not only the failed ones"
            )

            log("fail_regression")

        elif result < low:
            fail = True

            print(
                f"WIN: benchmark {key} failed, actual result {result} is {ratio:+.2f}% lower than "
                f"expected {entry.expected_value} ±{entry.noise_margin*100:.2f}% "
                f"please update the expected results. \n"
            )
            print(
                "please update all results that changed significantly, and not only the failed ones"
            )

            log("fail_win")

        else:
            print(
                f"PASS: benchmark {key} pass, actual result {result} {ratio:+.2f}% is within "
                f"expected {entry.expected_value} ±{entry.noise_margin*100:.2f}%\n"
            )

            log("pass")

    # Log all benchmarks that do not have a regression test enabled for them.
    for key, entry in result_data.items():
        if key not in expected_data:
            print(
                f"MISSING REGRESSION TEST: benchmark {key} does not have a regression test enabled for it.\n"
            )
            scribe.open_source_signpost(
                subsystem="pr_time_benchmarks",
                name="missing_regression_test",
                parameters=json.dumps(
                    {
                        "benchmark_name": entry.benchmark_name,
                        "metric_name": entry.metric_name,
                    }
                ),
            )

    with open(reference_expected_results_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for entry in new_expected.values():
            # Write the data to the CSV file
            # print(f"{entry.benchmark_name},{entry.metric_name,},{round(entry.expected_value)},{entry.noise_margin}")
            writer.writerow(
                [
                    entry.benchmark_name,
                    entry.metric_name,
                    entry.expected_value,
                    entry.noise_margin,
                ]
            )
            # Three empty rows for merge conflicts.
            writer.writerow([])
            writer.writerow([])
            writer.writerow([])

    print("new expected results file content if needed:")
    with open(reference_expected_results_path) as f:
        print(f.read())

    if fail:
        print(
            f"There was some failures you can use the new reference expected result stored at path:"
            f"{reference_expected_results_path} and printed above\n"
        )
        sys.exit(1)
    else:
        print("All benchmarks passed")


if __name__ == "__main__":
    main()
