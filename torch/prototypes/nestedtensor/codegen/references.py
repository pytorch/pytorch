import csv
from os import path


def _process_csv(refpath):
    refpath = path.join(path.dirname(__file__), refpath)
    rows = [x for x in csv.DictReader(open(refpath, 'r'))]
    return rows


def get_csv_functions():
    funs = _process_csv('./methods.csv')
    result = []
    for fun in funs:
        if (fun.get('pointwise unary', '') == 'TRUE'
            and not fun.get('inplace', False)
                and not fun.get('name', '').endswith('__')):
            result.append(fun['name'])
            print(fun['name'])
    return [
        'add',
        'mul',
        'sub',
        # 'div' it's truediv
        ]


def get_pointwise_comparison_functions():
    return [
        'eq',
        'ge',
        'gt',
        'le',
        'ne',
        'ge'
        ]
