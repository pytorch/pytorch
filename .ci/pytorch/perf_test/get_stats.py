import json
import sys

import numpy

sample_data_list = sys.argv[1:]
sample_data_list = [float(v.strip()) for v in sample_data_list]

sample_mean = numpy.mean(sample_data_list)
sample_sigma = numpy.std(sample_data_list)

data = {
    "mean": sample_mean,
    "sigma": sample_sigma,
}

print(json.dumps(data))
