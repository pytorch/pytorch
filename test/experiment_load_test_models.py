import torch
from torch.testing._internal.common_methods_invocations import op_db, OpInfo
from inspect import getmembers, isclass
from textwrap import dedent


import_template = "import torch\n"

model_template = dedent('''
    class Foo_{class_name}(torch.nn.Module):
        def __init__(self):
            super(Foo_{class_name}, self).__init__()

        def forward(self, inp):
            return torch.{op_name}(inp)

    foo_{class_name}_instance = torch.jit.script(Foo_{class_name}())
    ''')

end_template = dedent('''
    if __name__ == "__main__":
        for key, val in dict(globals()):
            print(key, val)
    ''')
# for op in op_db:
#     if isinstance(op, OpInfo):
#         is_errored = False
#         dtypes = op.supported_dtypes("cpu")
#         for dtype in dtypes:
#             inputs = op.sample_inputs(device="cpu", dtype=dtype)
#             for inp in inputs:
#                 try:
#                     op.get_op()(inp.input, *inp.args, **inp.kwargs)
#                 except RuntimeError as e:
#                     is_errored = True
#         if not is_errored:
#             possible_inputs = []
#             dtypes = op.supported_dtypes("cpu")
#             for dtype in dtypes:
#                 inputs_for_dtype = op.sample_inputs(device="cpu", dtype=dtype)
#                 possible_inputs.extend(inputs_for_dtype)

#             # # create the class object
#             # class_name = "TestOp" + op.name

#             # # forward method
#             # def forward(self, inp):
#             #     return self.op(inp)

#             # setattr(op.get_op(), "__globals__", [])

#             # op_class = type(class_name, (torch.nn.Module, ), {
#             #     "op": op.get_op(),
#             #     "op_name": op.name,
#             #     # member functions
#             #     "forward": forward,
#             # })

#             model_str = model_template.format(class_name=op.name, op_name=op.name)
#             exec(model_str)
#             s = globals["Foo" + op.name]
#             print(s)

#             # try:
#             #     scripted_class = torch.jit.script(op_class())
#             #     for inp in possible_inputs:
#             #         op_class()(inp.input, *inp.args, **inp.kwargs)

#             #     print(scripted_class.graph)
#             # except TypeError:
#             #     print(op_class().op_name)

def main():
    dump_file = import_template
    models = []
    for op in op_db:
        if isinstance(op, OpInfo):
            is_errored = False
            dtypes = op.supported_dtypes("cpu")
            for dtype in dtypes:
                inputs = op.sample_inputs(device="cpu", dtype=dtype)
                for inp in inputs:
                    try:
                        op.get_op()(inp.input, *inp.args, **inp.kwargs)
                    except RuntimeError as e:
                        is_errored = True
            if not is_errored:
                possible_inputs = []
                dtypes = op.supported_dtypes("cpu")
                for dtype in dtypes:
                    inputs_for_dtype = op.sample_inputs(device="cpu", dtype=dtype)
                    possible_inputs.extend(inputs_for_dtype)

                model_str = model_template.format(class_name=op.name.replace(".", "_"), op_name=op.name)
                models.append(model_str)

    models_str = "\n".join(models)

    dump_file += models_str
    dump_file += end_template

    with open("./test_models.py", "w") as f:
        f.write(dump_file)

if __name__ == "__main__":
    main()
