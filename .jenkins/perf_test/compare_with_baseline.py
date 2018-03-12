import sys
import json
import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test-name', dest='test_name', action='store',
                    required=True, help='test name')
parser.add_argument('--sample-stats', dest='sample_stats', action='store',
                    required=True, help='stats from sample')
parser.add_argument('--update', action='store_true',
                    help='whether to update baseline using stats from sample')
args = parser.parse_args()

test_name = args.test_name

if 'cpu' in test_name:
    backend = 'cpu'
elif 'gpu' in test_name:
    backend = 'gpu'

data_file_path = '../perf_test_numbers_{}.json'.format(backend)

with open(data_file_path) as data_file:
    data = json.load(data_file)

mean = float(data[test_name]['mean'])
sigma = float(data[test_name]['sigma'])

print("population mean: ", mean)
print("population sigma: ", sigma)

sample_stats_data = json.loads(args.sample_stats)

sample_mean = sample_stats_data['mean']
sample_sigma = sample_stats_data['sigma']

print("sample mean: ", sample_mean)
print("sample sigma: ", sample_sigma)

z_value = (sample_mean - mean) / sigma

print("z-value: ", z_value)

if z_value >= 2:
    raise Exception('''\n
z-value >= 2, there is >97.7% chance of perf regression.\n
To reproduce this regression, run `cd .jenkins/perf_test/ && bash ''' + test_name + '''.sh` on your local machine and compare the runtime before/after your code change.
''')
else:
    print("z-value < 2, no perf regression detected.")
    if args.update:
        print("We will use these numbers as new baseline.")
        new_data_file_path = '../new_perf_test_numbers_{}.json'.format(backend)
        with open(new_data_file_path) as new_data_file:
            new_data = json.load(new_data_file)
        new_data[test_name]['mean'] = sample_mean
        new_data[test_name]['sigma'] = sample_sigma
        with open(new_data_file_path, 'w') as new_data_file:
            json.dump(new_data, new_data_file, indent=4)
