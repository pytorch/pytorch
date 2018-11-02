from __future__ import absolute_import, division, print_function

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


with open(sys.argv[1]) as f:
    reader = csv.reader(f, delimiter=" ")
    for row in reader:
        tensor_name = row[3]
        minimum = float(row[4])
        maximum = float(row[5])
        nbins = int(row[6])

        bins = np.linspace(minimum, maximum, nbins)
        hist = row[7:]

        # y_pos = np.arrange(len(bins))

        plt.bar(bins, hist, bins[1] - bins[0])
        # plt.xticks(y_pos, bins)
        plt.savefig(tensor_name.replace("/", "_") + ".png")
        plt.close()
