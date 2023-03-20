from collections import defaultdict
import re 
import ast
import json
import pandas as pd
import matplotlib.pyplot as plt
bad_files = set()
def get_oncall_from_testfile(testfile: str):
    path = f"test/{testfile}"
    if not path.endswith(".py"):
        path += ".py"
    # get oncall on test file 
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("# Owner(s): "):
                        possible_lists = re.findall('\[.*\]', line)
                        if len(possible_lists) > 1:
                            raise Exception("More than one list found")
                        elif len(possible_lists) == 0:
                            raise Exception("No oncalls found or file is badly formatted")
                        oncalls = ast.literal_eval(possible_lists[0])
                        return oncalls
    except Exception as e:
        if "/" in test_file:
            return ["{testfile.split('/')[0]}"]
        else:
            bad_files.add(testfile)
            # print(f"bad_file: {testfile}")
    return None

def read_json_file(path):
    with open(path) as f:
        return json.load(f)

def read_json(path):
    with open(path) as f:
        return json.load(f)
    
def flatten_dict(nested_dict):
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res

if __name__ == '__main__':
    # read json file

    oncalls = read_json("oncall_list.json")
    prices = read_json("price_per_config.json")
    file_to_oncall = defaultdict(lambda: [])
    for oncall, files in oncalls.items():
        for file in files:
            file_to_oncall[file].append(oncall)

    team_to_time = defaultdict(lambda: 0)
    team_to_price = defaultdict(lambda: 0)
    test_times = read_json_file("test_times.json")
    flattened = flatten_dict(test_times)
    # df = pd.DataFrame.from_dict(test_times, orient="index")
    # print(df)
    oncalls_to_file = defaultdict(lambda: [])
    config_to_runner = defaultdict(lambda: defaultdict(lambda: 0))
    configs = set()
    modes = set()
    for config_name, config in test_times.items():
        for mode, file_to_time in config.items():
            for test_file, time in file_to_time.items():
                # tests are measured in seconds and are the sum of 3 runs
                # we want to convert to hours for a single run
                time = float(time) / 3 / 60 / 60
                oncalls = file_to_oncall[test_file]
                for oncall in oncalls:
                    team_to_time[oncall] += time
                    team_to_price[oncall] += time * prices[mode][config_name]
    df_times = pd.DataFrame.from_dict(team_to_time, orient="index", columns=["time (hours)"])
    df_times["time (percentage)"] = (df_times["time (hours)"] / df_times["time (hours)"].sum()) * 100
    df_costs = pd.DataFrame.from_dict(team_to_price, orient="index", columns=["cost (USD)"])
    df_costs["cost (percentage)"] = (df_costs["cost (USD)"] / df_costs["cost (USD)"].sum()) * 100
    df = pd.concat([df_times, df_costs], axis=1)
    df = df.sort_values(by=['time (percentage)'], ascending=False)
    graphable = df[df["time (percentage)"] > 6]
    not_graphable = df[df["time (percentage)"] <= 6]
    graphable.loc["Other"] = not_graphable.sum()
    print(df.to_csv())

    #  plotting code
    
    # def autopct_format(values):
    #     def my_format(pct):
    #         total = sum(values)
    #         val = int(round(pct*total/100.0))
    #         return '{:.1f}%\n({v:d})'.format(pct, v=val)
    #     return my_format

    # plt.pie(graphable["time (hours)"],labels = graphable.index, autopct=autopct_format(graphable["time (hours)"]))
    # plt.title("Time spent Running Tests per oncall (hours)", fontsize=20)
    # plt.show()
    # plt.pie(graphable["cost (USD)"],labels = graphable.index, autopct=autopct_format(graphable["cost (USD)"]))
    # plt.title("Cost of Running Tests per oncall (USD)", fontsize=20)
    # plt.show()