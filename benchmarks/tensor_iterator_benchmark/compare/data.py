import json
from collections import defaultdict
from typing import NamedTuple, Union, Tuple

def load(filename):
    with open(filename) as f:
        return json.load(f)

class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))

class DataItem(NamedTuple):
    problem_size: Union[int, Tuple[int, int]]
    result: float

def data_to_tuple(data):
    return sorted([DataItem(x['problem_size'], x['result']) for x in data])

def values_to_string(setup):
    return {k: str(v) for k, v in setup.items()}

def compare(baseline, new):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for title, experiment in baseline.items():
        for e in experiment:
            setup, data = values_to_string(e['setup']), e['data']
            result[title][hashabledict(setup)]['baseline'] = data_to_tuple(data)
    for title, experiment in new.items():
        for e in experiment:
            setup, data = values_to_string(e['setup']), e['data']
            result[title][hashabledict(setup)]['new'] = data_to_tuple(data)
    for title, experiment in result.items():
        for setup, data in experiment.items():
            for (problem_size1, result1), (problem_size2, result2) in zip(data['baseline'], data['new']):
                assert problem_size1 == problem_size2
                diff = result2 / result1 - 1
                data['compare'].append(DataItem(problem_size1, diff))
    return result
