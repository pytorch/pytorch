from collections import defaultdict
import re 
import ast
import json
import pandas as pd
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
    file_to_oncall = defaultdict(lambda: [])
    for oncall, files in oncalls.items():
        for file in files:
            file_to_oncall[file].append(oncall)

    team_to_times = defaultdict(lambda: 0)
    test_times = read_json_file("test_times.json")
    flattened = flatten_dict(test_times)
    # df = pd.DataFrame.from_dict(test_times, orient="index")
    # print(df)
    oncalls_to_file = defaultdict(lambda: [])
    configs = set()
    modes = set()
    for config_name, config in test_times.items():
        configs.add(config_name)
        for mode, file_to_time in config.items():
            modes.add(mode)
            for test_file, time in file_to_time.items():
                # print(time)
                time = float(time)
                oncalls = file_to_oncall[test_file]
                for oncall in oncalls:
                    team_to_times[(config_name, mode, oncall)] += time
    # print(team_to_times)
    final_dict = {i: [config_name, mode, oncall, time] for i, ((config_name, mode, oncall), time) in enumerate(team_to_times.items())}
    df = pd.DataFrame.from_dict(final_dict, orient="index", columns=["config_name", "mode", "oncall", "time"])
    df["time"] = df["time"] / 60 /  60 
    df = df.groupby(["mode", "oncall"])["time"].sum().reset_index()
    df = df.sort_values(by=["mode", 'time'])
    df = df[(df["time"] > 1)]
    # df["in_minutes"] = df["time"] / 60
    # df["in_hours"] = df["in_minutes"] / 60
    print(df.to_markdown())
    # print(configs)
    # print(modes)
    # pd.display(df)
    # print(oncalls_to_file)
    # print(json.dumps(oncalls_to_file, indent=4))
    # print(bad_files)
