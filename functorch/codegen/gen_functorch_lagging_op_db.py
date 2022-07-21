# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.testing._internal.common_methods_invocations import op_db


def num_leading_spaces(line: str) -> int:
    result = len(line) - len(line.lstrip())
    # Empty space handling
    if result == 0:
        return 999999
    return result


def deindent(code: str) -> str:
    lines = code.split('\n')
    min_leading_spaces = min(map(num_leading_spaces, lines))
    lines = [line[min_leading_spaces:] for line in lines]
    return '\n'.join(lines)


if __name__ == '__main__':
    supported = {(opinfo.name, opinfo.variant_test_name) for opinfo in op_db}
    supported = sorted(supported)
    print(deindent("""\
    # Copyright (c) Facebook, Inc. and its affiliates.
    # All rights reserved.
    #
    # This source code is licensed under the BSD-style license found in the
    # LICENSE file in the root directory of this source tree.
    from torch.testing._internal.common_methods_invocations import op_db

    # Generated from codegen/gen_functorch_op_db.py via
    # python codegen/gen_functorch_lagging_op_db.py > test/functorch_lagging_op_db.py
    #
    # People add new OpInfos to PyTorch all the time.
    # We want them to be able to add OpInfos without breaking our CI.
    # To achieve this, we keep our OpInfo library behind that of Pytorch's and
    # we periodically update our OpInfo library by regenerating this file"""))

    print("_functorch_lagging_meta = {")
    for name, variant in supported:
        print(f'    {(name, variant)},')
    print("}")

    print(deindent("""\


    def in_functorch_lagging_op_db(opinfo):
        return (opinfo.name, opinfo.variant_test_name) in _functorch_lagging_meta


    functorch_lagging_op_db = [
        opinfo for opinfo in op_db if in_functorch_lagging_op_db(opinfo)
    ]"""))
