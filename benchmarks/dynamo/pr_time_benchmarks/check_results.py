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

    # Read expected data file.
    expected_data: dict[str, ExpectedFileEntry] = {}

    with open(expected_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
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
    for key, entry in expected_data.items():
        if key not in result_data:
            print(f"Missing entry for {key} in result file")
            sys.exit(1)

        low = entry.expected_value - entry.expected_value * entry.noise_margin
        high = entry.expected_value + entry.expected_value * entry.noise_margin
        result = result_data[key].actual_value

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
                    }
                ),
            )

        if result > high:
            fail = True
            ratio = float(result - entry.expected_value) * 100 / entry.expected_value
            print(
                f"REGRESSION: benchmark {key} failed, actual result {result} "
                f"is {ratio:.2f}% higher than expected {entry.expected_value} ±{entry.noise_margin*100:.2f}% "
                f"if this is an expected regression, please update the expected results."
            )

            log("fail_regression")

        if result < low:
            fail = True
            ratio = float(entry.expected_value - result) * 100 / entry.expected_value

            print(
                f"WIN: benchmark {key} failed, actual result {result} is {ratio:.2f}% lower than "
                f"expected {entry.expected_value} ±{entry.noise_margin*100:.2f}% "
                f"please update the expected results."
            )

            log("fail_win")

    # Log all benchmarks that do not have a regression test enabled for them.
    for key, entry in result_data.items():
        if key not in expected_data:
            print(
                f"MISSING REGRESSION TEST: benchmark {key} does not have a regression test enabled for it"
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
    if fail:
        sys.exit(1)
    else:
        print("All benchmarks passed")


if __name__ == "__main__":
    main()
