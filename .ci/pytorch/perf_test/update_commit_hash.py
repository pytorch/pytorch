import json
import sys

data_file_path = sys.argv[1]
commit_hash = sys.argv[2]

with open(data_file_path) as data_file:
    data = json.load(data_file)

data["commit"] = commit_hash

with open(data_file_path, "w") as data_file:
    json.dump(data, data_file)
