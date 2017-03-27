import json
import torch
import torch.legacy.optim as optim
from pprint import pprint


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.DoubleTensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * x * (y - x ** 2)))

algorithms = {
    'adadelta': optim.adadelta,
    'adagrad': optim.adagrad,
    'adam': optim.adam,
    'adamax': optim.adamax,
    'asgd': optim.asgd,
    'cg': optim.cg,
    'nag': optim.nag,
    'rmsprop': optim.rmsprop,
    'rprop': optim.rprop,
    'sgd': optim.sgd,
    'lbfgs': optim.lbfgs,
}

with open('tests.json', 'r') as f:
    tests = json.loads(f.read())

for test in tests:
    print(test['algorithm'] + '\t')
    algorithm = algorithms[test['algorithm']]
    for config in test['config']:
        print('================================================================================\t')
        params = torch.DoubleTensor((1.5, 1.5))
        for i in range(100):
            algorithm(lambda x: (rosenbrock(x), drosenbrock(x)), params, config)
            print('{:.8f}\t{:.8f}\t'.format(params[0], params[1]))
