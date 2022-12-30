import ast
import typing
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--symbolic_sizes", nargs="+", type=int)
cmd_args = parser.parse_args()
symbolic_sizes_index = 0

input = sys.stdin.read()

lines = input.strip().splitlines()

is_aten = False


def parse(l: str):
    ast_ = ast.parse(l)
    assert len(ast_.body) == 1
    return ast_.body[0]


def replace_name(name: str):
    ops_prefix = "fd.ops."
    if name.startswith("T"):
        if is_aten:
            return f"t{name[1:]}"
        else:
            return f"tv{name[1:]}"
    elif name.startswith("S"):
        return f"s{name[1:]}"
    elif name.startswith(ops_prefix):
        op = name[len(ops_prefix):]
        if is_aten:
            return "at::" + op
        else:
            if op == "cast":
                return "castOp"
            if op == "var_mean":
                return "variance_mean"
            return op
    elif name == "fd.add_output":
        if is_aten:
            return "outputs.push_back"
        else:
            return "fusion->addOutput"
    elif name == "fd.define_constant":
        if not is_aten:
            return "IrBuilder::create"
    return name


def tocppstr(x):
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def list2vector(l: typing.Union[ast.List, list]):
    if isinstance(l, ast.List):
        l = eval(ast.unparse(l))
    l = [tocppstr(x) for x in l]
    l = "{" + ", ".join(l) + "}"
    return ast.Name(l)


def handle_call(l: ast.Call):
    func = ast.Name(replace_name(ast.unparse(l.func)))
    args = handle(l.args)
    keywords = l.keywords
    if func.id == "IrBuilder::create":
        assert len(args) == 1
        arg = args[0]
        assert isinstance(arg, ast.Constant)
        arg = arg.value
        if isinstance(arg, float):
            func.id = func.id + "<Double>"
        elif isinstance(arg, int):
            func.id = func.id + "<Int>"
    elif func.id == "fd.define_constant":
        assert is_aten
        arg = args[0]
        assert isinstance(arg, ast.Constant)
        return arg
    elif func.id == "castOp":
        assert len(args) == 1
        assert len(keywords) == 1
        keyword = keywords[0]
        assert isinstance(keyword, ast.keyword)
        assert keyword.arg == "dtype"
        value = ast.unparse(keyword.value).replace(".", "::")
        value = ast.Name(value)
        args.insert(0, value)
        keywords = []
    elif func.id == "at::cast":
        assert is_aten
        assert len(args) == 1
        assert len(keywords) == 1
        keyword = keywords[0]
        assert isinstance(keyword, ast.keyword)
        assert keyword.arg == "dtype"
        value = ast.unparse(keyword.value).replace("DataType.", "ScalarType::")
        value = ast.Name(value)
        func = ast.Attribute(args[0], "to")
        args[0] = value
        keywords = []
    elif func.id == "view" or func.id == "at::view":
        assert len(args) == 1
        assert len(keywords) == 2

        original_shape = keywords[0]
        assert original_shape.arg == "original_shape"
        original_shape = list2vector(original_shape.value)

        new_shape = keywords[1]
        assert new_shape.arg == "new_shape"
        new_shape = list2vector(new_shape.value)

        if is_aten:
            func = ast.Attribute(args[0], "view")
            args = [new_shape]
        else:
            args.extend([original_shape, new_shape])

        keywords = []
    elif func.id == "fd.define_tensor":
        assert len(keywords) == 3
        assert len(args) == 0

        symbolic_sizes = keywords[0]
        assert symbolic_sizes.arg == "symbolic_sizes"
        symbolic_sizes_val = symbolic_sizes.value
        ndims = len(symbolic_sizes_val.elts)
        symbolic_sizes = list2vector(symbolic_sizes_val)

        contiguous = keywords[1]
        assert contiguous.arg == "contiguous"
        contiguous = contiguous.value
        assert ndims == len(contiguous.elts)
        contiguous = list2vector(contiguous)

        dtype = keywords[2]
        assert dtype.arg == "dtype"
        dtype = ast.unparse(dtype.value)
        if is_aten:
            sizes = symbolic_sizes
            if cmd_args.symbolic_sizes is not None:
                sizes = eval(ast.unparse(symbolic_sizes_val))
                for i, s in enumerate(sizes):
                    if s == -1:
                        global symbolic_sizes_index
                        sizes[i] = cmd_args.symbolic_sizes[symbolic_sizes_index]
                        symbolic_sizes_index += 1
                sizes = list2vector(sizes)
            result = ast.Call(ast.Name("at::randn"), [sizes, ast.Name("options")], [])
            if dtype != "DataType.Float":
                to = ast.Attribute(result, "to")
                result = ast.Call(to, [ast.Name(dtype.replace("DataType.", "ScalarType::"))], [])
            if "false" in contiguous.id:
                contig = ast.Attribute(result, "set_contiguous")
                result = ast.Call(contig, [contiguous], [])
            return result
        else:
            builder = ast.Name("TensorViewBuilder()")
            ndims_call = ast.Call(ast.Attribute(builder, "ndims"), [ast.Constant(ndims)], [])
            shape_call = ast.Call(ast.Attribute(ndims_call, "shape"), [symbolic_sizes], [])
            contig_call = ast.Call(ast.Attribute(shape_call, "contiguity"), [contiguous], [])
            dtype_call = ast.Call(ast.Attribute(contig_call, "dtype"), [ast.Name(dtype.replace(".", "::"))], [])
            build_call = ast.Call(ast.Attribute(dtype_call, "build"), [], [])
            return build_call
    elif func.id == "broadcast_in_dim" or func.id == "at::broadcast_in_dim":
        assert len(keywords) == 2
        assert len(args) == 1

        output_shape = keywords[0]
        assert output_shape.arg == "output_shape"
        output_shape = output_shape.value
        output_shape = eval(ast.unparse(output_shape))

        broadcast_dims = keywords[1]
        assert broadcast_dims.arg == "broadcast_dims"
        broadcast_dims = broadcast_dims.value
        broadcast_dims = eval(ast.unparse(broadcast_dims))

        n_out_dims = len(output_shape)

        is_broadcast = [True] * n_out_dims
        for orig_dim in broadcast_dims:
            is_broadcast[orig_dim] = False

        if is_aten:
            result = args[0]
            for i, b in enumerate(is_broadcast):
                if b:
                    result = ast.Call(ast.Attribute(result, "unsqueeze"), [ast.Constant(i)], [])
            result = ast.Call(ast.Attribute(result, "expand"), [list2vector(output_shape)], [])
        else:
            result = ast.Call(ast.Name("broadcast"), [args[0], list2vector(is_broadcast)], [])
            result = ast.Call(ast.Name("expand"), [result, list2vector(
                [f"IrBuilder::create<Int>({x})" for x in output_shape])], [])

        return result
    elif func.id == "at::var_mean" or func.id == "variance_mean":
        assert len(args) == 1
        assert len(keywords) == 3

        axes = keywords[0]
        assert isinstance(axes, ast.keyword)
        assert axes.arg == "axes"
        axes = list2vector(axes.value)

        correction = keywords[1]
        assert isinstance(correction, ast.keyword)
        assert correction.arg == "correction"
        correction = correction.value

        keepdim = keywords[2]
        assert isinstance(keepdim, ast.keyword)
        assert keepdim.arg == "keepdim"
        keepdim = keepdim.value
        assert isinstance(keepdim, ast.Constant)
        keepdim = ast.Name(tocppstr(keepdim.value))

        args.extend([axes, correction, keepdim])
        keywords = []

    return ast.Call(func, args, keywords)


def handle(l):
    if isinstance(l, list):
        result = []
        for item in l:
            result.append(handle(item))
        return result
    elif isinstance(l, ast.Assign):
        create = ast.Assign(handle(l.targets), handle(l.value), l.type_comment)
        if len(create.targets) == 1 and isinstance(create.targets[0], ast.Tuple):
            targets = create.targets[0].elts
            tuple_name = ast.Name("_".join([x.id for x in targets]))
            create.targets[0] = tuple_name
            result = [create]
            for i, n in enumerate(targets):
                value = ast.Call(ast.Name(f"std::get<{i}>"), [tuple_name], [])
                result.append(ast.Assign([n], value))
            return result
        is_define_tensor = isinstance(l.value, ast.Call) and ast.unparse(l.value.func) == "fd.define_tensor"
        if is_define_tensor:
            if is_aten:
                func = "inputs.push_back"
            else:
                func = "fusion->addInput"
            assert len(create.targets) == 1
            add = ast.Call(ast.Name(func), [create.targets[0]], [])
            return [create, add]
        else:
            return create
    elif isinstance(l, ast.Name):
        return ast.Name(replace_name(l.id), l.ctx)
    elif isinstance(l, ast.Call):
        return handle_call(l)
    elif isinstance(l, ast.Expr):
        return ast.Expr(handle(l.value))
    elif isinstance(l, ast.Tuple):
        return ast.Tuple([handle(x) for x in l.elts])
    return l


def ast2str(l):
    if isinstance(l, ast.Assign):
        assert len(l.targets) == 1
        return f"auto {ast2str(l.targets[0])} = {ast2str(l.value)}"
    l.lineno = 0
    return ast.unparse(l)


test_str = """TEST_F(NVFuserTest, FusionGeneratedTest_CUDA) {
  std::unique_ptr<Fusion> fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  {
"""

for l in lines:
    l = parse(l)
    l = handle(l)
    if not isinstance(l, list):
        l = [l]
    for x in l:
        test_str += f"    {ast2str(x)};\n"

test_str += """  }

  auto options = at::TensorOptions().dtype(kFloat).device(at::kCUDA, 0);
  std::vector<IValue> inputs;
  std::vector<Tensor> outputs;

  {
"""

is_aten = True

for l in lines:
    l = parse(l)
    l = handle(l)
    if not isinstance(l, list):
        l = [l]
    for x in l:
        test_str += f"    {ast2str(x)};\n"

test_str += """  }

  FusionExecutorCache fec(std::move(fusion_ptr));
  auto cg_outputs = fec.runFusionWithInputs(inputs);
  testValidate(fusion, cg_outputs, inputs, outputs, __LINE__, __FILE__);
}"""

print(test_str)
