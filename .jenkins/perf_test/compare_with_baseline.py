import sys
import json
import numpy
from scipy import stats

with open('../perf_test_numbers.json') as data_file:
    data = json.load(data_file)

test_name = sys.argv[1]

mean = float(data[test_name]['mean'])
sigma = float(data[test_name]['sigma'])

print("population mean: ", mean)
print("population sigma: ", sigma)

sample_stats_data = json.loads(sys.argv[2])

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
