#!/usr/bin/python3
# imports for the benefit of documentation
import torch
import torch.nn.functional as F
import math
import numpy
# utility
import inspect
import re
import sys
import os
import ast
import astor
# import autopep8

# I only tested with python3
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO


def execute_and_get_output(cmd, locals):
    # only comment cannot be executed
    if not re.sub("#.*$", "", cmd).strip():
        return "", "", None
    s = StringIO()
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = s, s
    try:
        eval(compile(cmd, "", "single"), locals)
    except Exception as e:
        sys.stdout, sys.stderr = stdout, stderr
        if type(e) == SyntaxError and str(e).startswith("unexpected EOF"):
            return "", cmd, None
        return s.getvalue() + "\n" + str(e), "", e
    sys.stdout, sys.stderr = stdout, stderr
    return s.getvalue(), "", None


def check_torch_module():
    members = [(k, inspect.getdoc(v)) for k, v in inspect.getmembers(torch)
               if not inspect.ismodule(v) and not k.startswith('_')]
    no_docstring = [k for k, v in members if v is None]
    members = [(k, v) for k, v in members if v is not None]
    print("No docstring for", no_docstring)
    for name, doc in members:
        ok, suggested_docstring = reproduce_docstring(name, doc)
        if ok:
            print(name, "OK", file=sys.stderr)
        elif suggested_docstring:
            pass
            # print(name, "REPAIRABLE", file=sys.stderr)
            # print("could use\n-----------------")
            # print(suggested_docstring)
            # print("------------")
        else:
            print(name, "BROKEN", file=sys.stderr)


def reproduce_docstring(name, doc, verbose=True):
    locals = {'torch': torch, 'math': math, 'F': torch.nn.functional,
              'numpy': numpy}
    torch.manual_seed(0)
    output = []
    ok = True
    cmd = ""
    suggested_docstring = []
    needs_manual_fix = False
    last_cmd = None
    doc = doc.split('\n')
    for lineno, l in enumerate(doc):
        if l and not l[0].isspace():
            last_cmd = None
        expected = ''
        if output:
            expected = output.pop(0)
            while l != expected and not expected:
                expected = output.pop(0)
        if output or expected:
            if l != expected:
                if verbose:
                    print("mismatch after command {}:\n{}\n{}".format(last_cmd, repr(expected), repr(l)),
                          file=sys.stderr)
                ok = False
                suggested_docstring.append(expected)
            else:
                suggested_docstring.append(l)
        if cmd == "":
            m = re.match("(\s*)>>> (.*)", l)
            if m:
                spaces, cmd = m.groups()
            if m or (not output and (last_cmd is None or not l[:1].isspace())):
                if ((l or (suggested_docstring and suggested_docstring[-1]) or
                     len(doc) > lineno + 1 and doc[lineno + 1][:1] == '.')):
                    suggested_docstring.append(l)
        else:
            if not l[:len(spaces) + 4].isspace():
                if verbose:
                    print("found", repr(l[:len(spaces) + 4]),
                          "when looking for spaces in mulitline command, marking BROKEN",
                          file=sys.stderr)
                ok = False
                needs_manual_fix = True
            else:
                suggested_docstring.append(l)
            cmd = cmd + "\n" + l[len(spaces) + 4:]
        if cmd:
            last_cmd = cmd
            res, cmd, exception = execute_and_get_output(cmd, locals)
            if exception:
                if verbose:
                    print("exception {}, marking BROKEN".format(exception), file=sys.stderr)
                needs_manual_fix = True
            if cmd == "":
                if res is not None and res != '':
                    output = [(spaces + l).rstrip() for l in str(res).split("\n")]
                    while output and not output[-1]:
                        del output[-1]
                else:
                    output = []
    if output and [l for l in output if l != '']:
        if verbose:
            print("remaining output after{}:".format(name), output, file=sys.stderr)
        ok = False
        while output and output[-1] == '':
            del output[-1]
        suggested_docstring += output
    if needs_manual_fix:
        return ok, None
    else:
        if suggested_docstring[0]:
            suggested_docstring.insert(0, "")
        return ok, "\n".join(suggested_docstring)


def joinattr(a):
    if type(a) == ast.Name:
        return a.id
    elif type(a) == ast.Attribute:
        return joinattr(a.value) + '.' + a.attr


class MyVisitor(ast.NodeTransformer):
    def visit_Call(self, node):
        global anode
        anode = node
        if getattr(getattr(node, 'func', None), 'id', None) == 'add_docstr':
            documented_function = joinattr(node.args[0])
            print(documented_function, node.args)
            docstring_node = None
            if type(node.args[1]) == ast.Str:
                docstring_node = node.args[1]
            elif (type(node.args[1]) == ast.Call and
                  type(node.args[1].func) == ast.Attribute and
                  type(node.args[1].func.value) == ast.Str):
                docstring_node = node.args[1].func.value
            else:
                import pdb; pdb.set_trace()
                ast.BinOp
                raise Exception("funny " + documented_function + ":" + ast.dump(node))
            if docstring_node is not None:
                docstring = docstring_node.s
                if docstring.endswith("\nFIXME"):
                    docstring = docstring[:-len("FIXME")]
                    docstring_node.s = docstring
                ok, suggested = reproduce_docstring(documented_function, docstring)
                if ok:
                    print(documented_function, "OK", file=sys.stderr)
                elif suggested is not None:
                    print(documented_function, "REPAIRED", file=sys.stderr)
                    docstring_node.s = suggested
                else:
                    print(documented_function, "BROKEN", file=sys.stderr)
                    if not docstring.endswith("FIXME"):
                        docstring_node.s = docstring + "\nFIXME"
        return node


def my_pretty_string(s, embedded, current_line, uni_lit=False, min_trip_str=20, max_line=100):
    r = astor.string_repr.pretty_string(s, embedded, current_line, uni_lit=uni_lit,
                                        min_trip_str=min_trip_str, max_line=max_line)
    if r.startswith('"""') and '\n' in r or '\\' in r:
        fancy = 'r"""' + s + '"""'
        if eval(fancy) == s and '\r' not in fancy:
            return fancy
    return r


def process_file():
    torch_docs_fn = 'torch/regenerate_docs.py'
    # torch_docs_fn = os.path.join(os.path.dirname(torch.__file__), 'regenerate_docs.py')
    txt = open(torch_docs_fn).read()
    tree = ast.parse(txt)
    MyVisitor().visit(tree)
    src1 = astor.to_source(tree, pretty_string=my_pretty_string)
    # src2 = autopep8.fix_code(src1)
    print(src1, end="")


# call without arguments to iterate through docstrings in torch (test mode)
# call with "--file" to produce a regenerated torch/regenerate_docs.py
if __name__ == "__main__":
    process_file()
    # if len(sys.argv) > 1 and sys.argv[1] == '--file':
    #     process_file()
    # else:
    #     check_torch_module()
