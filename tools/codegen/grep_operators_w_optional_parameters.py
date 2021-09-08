import argparse
import yaml

from tools.codegen.gen import get_grouped_native_functions, parse_native_yaml
from tools.codegen.gen_backend_stubs import parse_backend_yaml
from tools.codegen.model import Argument, OptionalType
from tools.codegen.utils import YamlLoader
from typing import Optional, Tuple

def has_optional_parameters(arguments: Tuple[Argument, ...]) -> bool:
    for argument in arguments:
        if type(argument.type) is OptionalType:
            return True

    return False


def construct_filter_set(backend_yaml_path: str) -> set:
    #TODO(jwtan): how about autograd?
    with open(backend_yaml_path, 'r') as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)

    supported = yaml_values.pop('supported', [])
    return set(supported)


def main() -> None:
    parser = argparse.ArgumentParser(description='Grep native functions that have operational parameters. Please run this script in the repository root.')
    parser.add_argument(
        '-f',
        '--filter',
        help='the ?_native_functions.yaml supported by different backends')
    options = parser.parse_args()

    native_functions = parse_native_yaml('aten/src/ATen/native/native_functions.yaml').native_functions
    total_operators = len(native_functions)

    filter_set = {}
    if options.filter is not None:
        filter_set = construct_filter_set(options.filter)
        total_operators = len(filter_set)

    print('Total operators: {}\n'.format(total_operators))

    result_functions = []
    for function in native_functions:
      if len(filter_set) > 0 and (str(function.func.name) not in filter_set):
        continue
      # TODO(jwtan): Examine the remaining attributes.
      if has_optional_parameters(function.func.arguments.pre_self_positional):
        result_functions.append(str(function.func))
        continue
      if has_optional_parameters(function.func.arguments.post_self_positional):
        result_functions.append(str(function.func))
        continue

    result_functions.sort()

    print('The following functions, total {}, have optional parameters:\n'.format(len(result_functions)))
    for function in result_functions:
        print(function)


if __name__ == '__main__':
    main()
