import argparse
import json
import math
import sys

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test-name", dest="test_name", action="store", required=True, help="test name"
)
parser.add_argument(
    "--sample-stats",
    dest="sample_stats",
    action="store",
    required=True,
    help="stats from sample",
)
parser.add_argument(
    "--update",
    action="store_true",
    help="whether to update baseline using stats from sample",
)
args = parser.parse_args()

test_name = args.test_name

if "cpu" in test_name:
    backend = "cpu"
elif "gpu" in test_name:
    backend = "gpu"

data_file_path = f"../{backend}_runtime.json"

with open(data_file_path) as data_file:
    data = json.load(data_file)

if test_name in data:
    mean = float(data[test_name]["mean"])
    sigma = float(data[test_name]["sigma"])
else:
    # Let the test pass if baseline number doesn't exist
    mean = sys.maxsize
    sigma = 0.001

print("population mean: ", mean)
print("population sigma: ", sigma)

# Let the test pass if baseline number is NaN (which happened in
# the past when we didn't have logic for catching NaN numbers)
if math.isnan(mean) or math.isnan(sigma):
    mean = sys.maxsize
    sigma = 0.001

sample_stats_data = json.loads(args.sample_stats)

sample_mean = float(sample_stats_data["mean"])
sample_sigma = float(sample_stats_data["sigma"])

print("sample mean: ", sample_mean)
print("sample sigma: ", sample_sigma)

if math.isnan(sample_mean):
    raise Exception("""Error: sample mean is NaN""")
elif math.isnan(sample_sigma):
    raise Exception("""Error: sample sigma is NaN""")

z_value = (sample_mean - mean) / sigma

print("z-value: ", z_value)

if z_value >= 3:
    raise Exception(
        f"""\n
z-value >= 3, there is high chance of perf regression.\n
To reproduce this regression, run
`cd .ci/pytorch/perf_test/ && bash {test_name}.sh` on your local machine
and compare the runtime before/after your code change.
"""
    )
else:
    print("z-value < 3, no perf regression detected.")
    if args.update:
        print("We will use these numbers as new baseline.")
        new_data_file_path = f"../new_{backend}_runtime.json"
        with open(new_data_file_path) as new_data_file:
            new_data = json.load(new_data_file)
        new_data[test_name] = {}
        new_data[test_name]["mean"] = sample_mean
        new_data[test_name]["sigma"] = max(sample_sigma, sample_mean * 0.1)
        with open(new_data_file_path, "w") as new_data_file:
            json.dump(new_data, new_data_file, indent=4)
