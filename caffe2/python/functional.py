from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from caffe2.python import core, workspace
from collections import namedtuple
from six import string_types

OpSchema = workspace.C.OpSchema


def namedtupledict(typename, field_names, *args, **kwargs):
    field_names_map = {n: i for i, n in enumerate(field_names)}
    # Some output names are invalid python identifier, e.g. "0"
    kwargs.setdefault('rename', True)
    data = namedtuple(typename, field_names, *args, **kwargs)

    def getitem(self, key):
        if isinstance(key, string_types):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)

    data.__getitem__ = getitem
    return data


class _Functional(object):
    def __getattribute__(self, op_type):
        def op_func(*inputs, **args):
            ws = workspace.C.Workspace()
            schema = OpSchema.get(op_type)
            input_prefix = 'input_'
            output_prefix = 'output_'

            def get_name_list(prefix, num, max_num):
                return [prefix + str(x) for x in range(min(num, max_num))]

            input_names, output_names = [], []
            input_names = get_name_list(
                input_prefix, len(inputs), schema.max_input
            )
            # verify the length of input name is in range
            # of schema
            num_input = len(input_names)
            if num_input > schema.max_input or num_input < \
               schema.min_input or not schema.num_inputs_allowed(num_input):
                raise ValueError(
                    "Functional C2: Number of inputs not in \
                range: {} - {} or not allowed."
                    .format(schema.min_input, schema.max_input)
                )

            if 'num_output' in args:
                num_output = args['num_output']
                if num_output > schema.max_output or \
                   num_output < schema.min_output or \
                   not schema.num_outputs_allowed(num_output) or \
                   not schema.num_inputs_outputs_allowed(num_input,
                                                         num_output):
                    raise ValueError(
                        "Functional C2: Number of output \
                    not in range: {} - {} or not allowed"
                        .format(schema.min_output, schema.max_output)
                    )
                output_names = get_name_list(
                    output_prefix, num_output, schema.max_output
                )
                args.pop('num_output')
            calculated = schema.CalculateOutput(num_input)
            if not output_names and calculated != -1:
                output_names = get_name_list(
                    output_prefix, calculated, schema.max_output
                )

            if not output_names:
                max_output = schema.max_output
                # For an op with max_output == inf
                # and no Output defined in schema
                # user should pass output_size explicitly
                if schema.inf == max_output:
                    raise ValueError(
                        "For operators with max_output == inf,\
                        user should pass num_output explicity."
                    )
                output_names = get_name_list(
                    output_prefix, max_output, max_output
                )
            for i, input_blob in enumerate(inputs):
                ws.create_blob(input_names[i]).feed(input_blob)

            op = core.CreateOperator(
                op_type, input_names, output_names, **args
            )
            ws._run_operator(op.SerializeToString())
            # RunOperator
            output_values = [ws.fetch_blob(x) for x in output_names]
            return namedtupledict('output', output_names)(*output_values)

        return op_func


Functional = _Functional()
