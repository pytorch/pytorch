# %%
import csv
import json
import pprint

cpu_time = "CPU time Us (sum)"


def load_shapes(filename):
    with open(filename, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        scuba_data = list(reader)

    for item in scuba_data:
        item["input_shapes"] = convert_to_tuple(json.loads(item["input_shapes"]))

    return scuba_data


def linear_shapes():
    raw_data = load_shapes("scuba-data-2021-09-10_linear.csv")
    filtered_data = []
    for row in raw_data:
        shape = row["input_shapes"]
        if len(shape) != 3:
            continue
        if len(shape[0]) < 2 or len(shape[1]) < 2:
            continue
        if len(shape[2]) != 1:
            continue
        filtered_data.append(row)
    return filtered_data


# Recursively converts lists to tuples
def convert_to_tuple(data):
    if isinstance(data, list):
        return tuple(convert_to_tuple(item) for item in data)
    else:
        return data


pprint.pprint(linear_shapes()[0])

# %%


# What stats should I make

from pandas import DataFrame

import os

files = [f for f in os.listdir(".") if "vs" in f]
pprint.pprint(files)
files = [
    "linear_matmul_cuda.json",
    "linear_matmul_fp16_cuda.json",
    # "linear_orig_mm_vs_orig_1cpu.json",
    # "linear_mkldnn_mm_vs_mlkdnn_1cpu.json",
    # "linear_mkldnn_1conv_vs_orig_1cpu.json",
    "linear_mkldnn_1conv_vs_orig_cpu.json",
    # "linear_mkldnn_mm_vs_orig_mm_1cpu.json",
    "linear_mkldnn_mm_vs_orig_1cpu.json",
    "linear_mkldnn_mm_vs_orig_cpu.json",
]

linear_res = linear_shapes()
linear_shape_lookup = {l["input_shapes"]: l for l in linear_res}


for file in files:
    with open(file) as f:
        data = json.load(f)

    for item in data:
        matching_item = linear_shape_lookup[convert_to_tuple(item["input_size"])]
        item["weight"] = matching_item[cpu_time]

    df = DataFrame.from_dict(data)
    df = df[["weight", "speedup"]]
    df = df.astype(float)
    df["ones"] = 1
    df["autotune_speedup"] = df[["speedup", "ones"]].max(axis=1)

    sum_weights = df.weight.sum()
    num_rows = df.shape[0]

    def linear_and_weighted(data: DataFrame, col_name, result_name):
        linear = data[col_name].sum() / num_rows
        weighted = (data[col_name] * data.weight).sum() / sum_weights
        print(f"{result_name}: {linear:1.4f}, weighted: {weighted:1.4f}")

    # All statistics both weighted and unweighted
    # % of ops 10% faster or slower
    # % of ops 1% faster or slower
    # % of ops with any speedup
    # Mean speedup
    print()
    print(file)
    linear_and_weighted(df, "speedup", r"mean speedup")
    linear_and_weighted(df, "autotune_speedup", r"mean speedup with autotuning")
    print("Percent of shapes that are X")
    linear_and_weighted(df[df.speedup > 1.1], "ones", r"10% faster")
    linear_and_weighted(df[df.speedup < 0.9], "ones", r"10% slower")
    # linear_and_weighted(df[df.speedup > 1.01], "ones", r"1% faster")
    # linear_and_weighted(df[df.speedup < 0.99], "ones", r"1% slower")
    linear_and_weighted(df[df.speedup > 1], "ones", r"any faster")

# %%


# %%
(df["ones"] * df.weight).sum()
# %%
