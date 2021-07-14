import libcst as cst
import libcst.matchers as m
import torch
import monkeytype
import test_model
import sys
from monkeytype.cli import apply_stub_using_libcst
from monkeytype.cli import get_stub
from collections import namedtuple
from torch import Tensor
import shutil

from monkeytype.stubs import (
    ExistingAnnotationStrategy,
    Stub,
    build_module_stubs_from_traces,
)

# a plumbing type to be able to apply annotations from MonkeyType
Args = namedtuple('Args', ['config', 'module_path', 'verbose', 'limit', 'disable_type_rewriting', 'existing_annotation_strategy', 'sample_count'])

COUNTER=0
#  tree nodes for outlining function calls 
ret_stmt = cst.parse_statement("return (a, b, c)")
call_stmt = cst.parse_statement(f"f{COUNTER}(val)")
call_expr =  call_stmt.body[0].value
call_arg =  call_stmt.body[0].value.args[0]
ann_stmt = cst.parse_statement(f"""
@torch.jit.ignore
def ann_func(a):
    pass
""")
ignore_ann = ann_stmt.decorators[0]

# trace fun from module `modname` for given `fun_args` , `kwargs`
def trace_types(fun, modname, *fun_args, **kwargs):
    with monkeytype.trace():
        fun(*fun_args, **kwargs)

    with open(f"{modname}.py") as f:
        jit_prog = f.read()
        
        args = Args(monkeytype.config.DefaultConfig(), (modname, ''), False, -1, False, ExistingAnnotationStrategy.REPLICATE, False)
        stub = get_stub(args, sys.stdout, sys.stderr)
        jit_prog = apply_stub_using_libcst(stub.render(), jit_prog, False)
        return jit_prog

    return None

def extract_main_method(jit_prog):
    init_tree = cst.parse_module(jit_prog)
    fun_defs = m.findall(init_tree, m.FunctionDef(name=cst.Name("add")))
    extract_mod = cst.Module(body=[fun_defs[0]])
    return extract_mod.code

def compile_str_get_error(ext_jit_prog):
    error = (False, -1)
    try:
        #cu = torch.jit.CompilationUnit(ext_jit_prog, _frames_up=0)
        torch.jit.script(ext_jit_prog)
    except Exception as e:
        e_str = str(e)
        error=(True, int(e_str[e_str.find("line")+5:e_str.find("\n", e_str.find("line"))]))

    return error

def create_call(args, returns):
    global COUNTER
    func_suffix = COUNTER
    call_stmt = cst.parse_statement(f"(r1, r2) = f{COUNTER}(val)")
    COUNTER += 1
    call_with_args = m.replace(call_stmt, m.Call(), lambda c, e : c.with_changes(args=args))
    call_with_args_returns = m.replace(call_with_args, m.Tuple(), lambda a, e: a.with_changes(elements=returns))
    return (func_suffix, call_with_args_returns)

def within_range(item, ranje):
    return ranje.start.line <= item.start.line and item.start.line <= ranje.end.line

def before(item, ranje):
    return item.start.line < ranje.start.line

def after(item, ranje):
    return item.start.line > ranje.end.line

class NameCollector(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ScopeProvider, cst.metadata.PositionProvider)
    def __init__(self):
        self.names = set()
        self.a_nodes = set()
        self.cur_block = None
        self.block_with_error = None
        self.line_with_error = None
        self.funcs = []
    
    # TODO: there are other types of statements that TS couldn't handle
    def visit_SimpleStatementLine(self, node):
        pos = self.get_metadata(cst.metadata.PositionProvider, node)
        print("visit visit_SimpleStatementLine ", id(node), " at ", pos.start.line)
        print("start=", pos.start)
        print("end=", pos.end)
        if error[0] and error[1] == pos.start.line:
            self.block_with_error = self.cur_block
            self.line_with_error = node
            print("setting block with error to ", id(self.cur_block))

    def visit_IndentedBlock(self, node):
        func_scope = self.get_metadata(cst.metadata.ScopeProvider, node)
        pos = self.get_metadata(cst.metadata.PositionProvider, node)
        self.cur_block = node
        print("visit Idented block ", id(node), " at ", pos.start.line)

    def leave_IndentedBlock(self, old_node, new_node):
        if old_node == self.block_with_error:
            func_scope = self.get_metadata(cst.metadata.ScopeProvider, old_node)
            pos = self.get_metadata(cst.metadata.PositionProvider, old_node)

            arg_names = set()
            for acc in func_scope.accesses:
                acc_pos = self.get_metadata(cst.metadata.PositionProvider, acc.node)
                print(f"{acc.node} at {acc_pos.start.line}")

                if within_range(acc_pos, pos):
                    if len(acc.referents) == 1:
                        # ignore builtin assignments (e.g. Import)
                        if hasattr(list(acc.referents)[0], "node"):
                            assgn_node = list(acc.referents)[0].node
                            acc_assgn_pos = self.get_metadata(cst.metadata.PositionProvider, assgn_node)
                            if within_range(acc_assgn_pos, pos):
                                continue

                        # ignore builtin definitions
                        if isinstance(list(acc.referents)[0], cst.metadata.BuiltinAssignment):
                            continue

                        print(f"needs to be captured: {acc.node} at {acc_pos.start.line}")
                        arg_names.add(acc.node.value)


            args = []
            for n in arg_names:
                new_arg = call_arg.with_changes(value=cst.Name(n))
                args.append(new_arg)


            # treat any assignment in the outlined block as a return
            # this is conservative and it's fine if it's not used in a caller
            # TODO: ignore builtin assignments
            returns = []
            for assgn in func_scope.assignments:
                print("assgn: ", assgn.name)
                assgn_pos = self.get_metadata(cst.metadata.PositionProvider, assgn.node)
                if within_range(assgn_pos, pos):
                    returns.append(cst.Element(value=cst.Name(value=assgn.name)))

            func_suffix, func_body = create_call(args, returns)

            func_ret_stmt = m.replace(ret_stmt, m.Tuple(), lambda a, e: a.with_changes(elements=returns))
            old_body_ret = cst.IndentedBlock(body=list(old_node.body) + [func_ret_stmt])
            self.funcs.append((func_suffix, old_body_ret, arg_names))
            return new_node.with_changes(body=[func_body])

        return new_node


# make a copy to try to compile iteratively
shutil.copy('test_model.py', 'test_model2.py')

i = 0
while True:

    import test_model2
    jit_prog = trace_types(test_model2.add, 'test_model2', torch.rand(2,2) * 1000, torch.rand(2,2) * 1000)
    t2 = open("test_model2.py", "w")
    t2.write('import torch\n')
    t2.write('from torch import Tensor\n')
    t2.write(jit_prog)
    t2.close()

    # for now we are trying to compile one main function or entry point to a module
    #ext_jit_prog = extract_main_method(jit_prog)


    # re-import with annotations
    import test_model2
    error = compile_str_get_error(test_model2.add)
    print(f"Error:{error}")
    # we were able to compile a method successfully
    if (not error[0]):
        break

    tree = cst.parse_module(jit_prog)
    wrapper = cst.MetadataWrapper(tree)
    nc = NameCollector()
    out = wrapper.visit(nc)
    # create outlined functions
    helper_defs = [cst.FunctionDef(name=cst.Name(f"f{x[0]}"), decorators=[ignore_ann], body=x[1], params=cst.Parameters(params=[cst.Param(cst.Name(value=y)) for y in x[2]])) for x in nc.funcs]
    # create a new module with outlined functions
    out = cst.Module(body=helper_defs + list(out.body))


    t2 = open("test_model2.py", "w")
    t2.write('import torch\n')
    t2.write('from torch import Tensor\n')
    t2.write(out.code)
    t2.close()

    if i >= 1:
        break
    i += 1
    
    # import test_model2
    # jit_prog = trace_types(test_model2.add, 'test_model2', torch.rand(2, 2) * 10000, torch.rand(2,2) * 10000)
    # print(jit_prog)

    # t2 = open("test_model2.py", "w")
    # t2.write(jit_prog)
    # t2.close()


    # print("done!")
    # torch.jit.script(test_model2.add)
