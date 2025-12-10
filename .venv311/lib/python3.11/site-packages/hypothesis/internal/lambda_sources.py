# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import ast
import hashlib
import inspect
import linecache
import sys
import textwrap
from collections.abc import Callable, MutableMapping
from inspect import Parameter
from typing import Any
from weakref import WeakKeyDictionary

from hypothesis.internal import reflection
from hypothesis.internal.cache import LRUCache

# we have several levels of caching for lambda descriptions.
# * LAMBDA_DESCRIPTION_CACHE maps a lambda f to its description _lambda_description(f).
#   Note that _lambda_description(f) may not be identical to f as it appears in the
#   source code file.
# * LAMBDA_DIGEST_DESCRIPTION_CACHE maps _function_key(f) to _lambda_description(f).
#   _function_key implements something close to "ast equality":
#   two syntactically identical (minus whitespace etc) lambdas appearing in
#   different files have the same key. Cache hits here provide a fast path which
#   avoids ast-parsing syntactic lambdas we've seen before. Two lambdas with the
#   same _function_key will not have different _lambda_descriptions - if
#   they do, that's a bug here.
# * AST_LAMBDAS_CACHE maps source code lines to a list of the lambdas found in
#   that source code. A cache hit here avoids reparsing the ast.
LAMBDA_DESCRIPTION_CACHE: MutableMapping[Callable, str] = WeakKeyDictionary()
LAMBDA_DIGEST_DESCRIPTION_CACHE: LRUCache[tuple[Any], str] = LRUCache(max_size=1000)
AST_LAMBDAS_CACHE: LRUCache[tuple[str], list[ast.Lambda]] = LRUCache(max_size=100)


def extract_all_lambdas(tree):
    lambdas = []

    class Visitor(ast.NodeVisitor):

        def visit_Lambda(self, node):
            lambdas.append(node)
            self.visit(node.body)

    Visitor().visit(tree)
    return lambdas


def extract_all_attributes(tree):
    attributes = []

    class Visitor(ast.NodeVisitor):
        def visit_Attribute(self, node):
            attributes.append(node)
            self.visit(node.value)

    Visitor().visit(tree)
    return attributes


def _function_key(f, *, bounded_size=False, ignore_name=False):
    """Returns a digest that differentiates functions that have different sources.

    Either a function or a code object may be passed. If code object, default
    arg/kwarg values are not recoverable - this is the best we can do, and is
    sufficient for the use case of comparing nested lambdas.
    """
    try:
        code = f.__code__
        defaults_repr = repr((f.__defaults__, f.__kwdefaults__))
    except AttributeError:
        code = f
        defaults_repr = ()
    consts_repr = repr(code.co_consts)
    if bounded_size:
        # Compress repr to avoid keeping arbitrarily large strings pinned as cache
        # keys. We don't do this unconditionally because hashing takes time, and is
        # not necessary if the key is used just for comparison (and is not stored).
        if len(consts_repr) > 48:
            consts_repr = hashlib.sha384(consts_repr.encode()).digest()
        if len(defaults_repr) > 48:
            defaults_repr = hashlib.sha384(defaults_repr.encode()).digest()
    return (
        consts_repr,
        defaults_repr,
        code.co_argcount,
        code.co_kwonlyargcount,
        code.co_code,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        ignore_name or code.co_name,
    )


class _op:
    # Opcodes, from dis.opmap. These may change between major versions.
    NOP = 9
    LOAD_FAST = 85
    LOAD_FAST_LOAD_FAST = 88
    LOAD_FAST_BORROW = 86
    LOAD_FAST_BORROW_LOAD_FAST_BORROW = 87


def _normalize_code(f, l):
    # A small selection of possible peephole code transformations, based on what
    # is actually seen to differ between compilations in our test suite. Each
    # entry contains two equivalent opcode sequences, plus a condition
    # function called with their respective oparg sequences, which must return
    # true for the transformation to be valid.
    Checker = Callable[[list[int], list[int]], bool]
    transforms: tuple[list[int], list[int], Checker | None] = [
        ([_op.NOP], [], lambda a, b: True),
        (
            [_op.LOAD_FAST, _op.LOAD_FAST],
            [_op.LOAD_FAST_LOAD_FAST],
            lambda a, b: a == [b[0] >> 4, b[0] & 15],
        ),
        (
            [_op.LOAD_FAST_BORROW, _op.LOAD_FAST_BORROW],
            [_op.LOAD_FAST_BORROW_LOAD_FAST_BORROW],
            lambda a, b: a == [b[0] >> 4, b[0] & 15],
        ),
    ]
    # augment with converse
    transforms += [
        (
            ops_b,
            ops_a,
            condition and (lambda a, b, condition=condition: condition(b, a)),
        )
        for ops_a, ops_b, condition in transforms
    ]

    # Normalize equivalent code. We assume that each bytecode op is 2 bytes,
    # which is the case since Python 3.6. Since the opcodes values may change
    # between version, there is a risk that a transform may not be equivalent
    # -- even so, the risk of a bad transform producing a false positive is
    # minuscule.
    co_code = list(l.__code__.co_code)
    f_code = list(f.__code__.co_code)

    def alternating(code, i, n):
        return code[i : i + 2 * n : 2]

    i = 2
    while i < max(len(co_code), len(f_code)):
        # note that co_code is mutated in loop
        if i < min(len(co_code), len(f_code)) and f_code[i] == co_code[i]:
            i += 2
        else:
            for op1, op2, condition in transforms:
                if (
                    op1 == alternating(f_code, i, len(op1))
                    and op2 == alternating(co_code, i, len(op2))
                    and condition(
                        alternating(f_code, i + 1, len(op1)),
                        alternating(co_code, i + 1, len(op2)),
                    )
                ):
                    break
            else:
                # no point in continuing since the bytecodes are different anyway
                break
            # Splice in the transform and continue
            co_code = (
                co_code[:i] + f_code[i : i + 2 * len(op1)] + co_code[i + 2 * len(op2) :]
            )
            i += 2 * len(op1)

    # Normalize consts, in particular replace any lambda consts with the
    # corresponding const from the template function, IFF they have the same
    # source key.

    f_consts = f.__code__.co_consts
    l_consts = l.__code__.co_consts
    if len(f_consts) == len(l_consts) and any(
        inspect.iscode(l_const) for l_const in l_consts
    ):
        normalized_consts = []
        for f_const, l_const in zip(f_consts, l_consts, strict=True):
            if (
                inspect.iscode(l_const)
                and inspect.iscode(f_const)
                and _function_key(f_const) == _function_key(l_const)
            ):
                # If the lambdas are compiled from the same source, make them be the
                # same object so that the toplevel lambdas end up equal. Note that
                # default arguments are not available on the code objects. But if the
                # default arguments differ then the lambdas must also differ in other
                # ways, since default arguments are set up from bytecode and constants.
                # I.e., this appears to be safe wrt false positives.
                normalized_consts.append(f_const)
            else:
                normalized_consts.append(l_const)
    else:
        normalized_consts = l_consts

    return l.__code__.replace(
        co_code=bytes(co_code),
        co_consts=tuple(normalized_consts),
    )


_module_map: dict[int, str] = {}


def _mimic_lambda_from_node(f, node):
    # Compile the source (represented by an ast.Lambda node) in a context that
    # as far as possible mimics the context that f was compiled in. If - and
    # only if - this was the source of f then the result is indistinguishable
    # from f itself (to a casual observer such as _function_key).
    f_globals = f.__globals__.copy()
    f_code = f.__code__
    source = ast.unparse(node)

    # Install values for non-literal argument defaults. Thankfully, these are
    # always captured by value - so there is no interaction with the closure.
    if f.__defaults__:
        for f_default, l_default in zip(
            f.__defaults__, node.args.defaults, strict=True
        ):
            if isinstance(l_default, ast.Name):
                f_globals[l_default.id] = f_default
    if f.__kwdefaults__:  # pragma: no cover
        for l_default, l_varname in zip(
            node.args.kw_defaults, node.args.kwonlyargs, strict=True
        ):
            if isinstance(l_default, ast.Name):
                f_globals[l_default.id] = f.__kwdefaults__[l_varname.arg]

    # CPython's compiler treats known imports differently than normal globals,
    # so check if we use attributes from globals that are modules (if so, we
    # import them explicitly and redundantly in the exec below)
    referenced_modules = [
        (local_name, module)
        for attr in extract_all_attributes(node)
        if (
            isinstance(attr.value, ast.Name)
            and (local_name := attr.value.id)
            and inspect.ismodule(module := f_globals.get(local_name))
        )
    ]

    if not f_code.co_freevars and not referenced_modules:
        compiled = eval(source, f_globals)
    else:
        if f_code.co_freevars:
            # We have to reconstruct a local closure. The closure will have
            # the same values as the original function, although this is not
            # required for source/bytecode equality.
            f_globals |= {
                f"__lc{i}": c.cell_contents for i, c in enumerate(f.__closure__)
            }
            captures = [f"{name}=__lc{i}" for i, name in enumerate(f_code.co_freevars)]
            capture_str = ";".join(captures) + ";"
        else:
            capture_str = ""
        if referenced_modules:
            # We add import statements for all referenced modules, since that
            # influences the compiled code. The assumption is that these modules
            # were explicitly imported, not assigned, in the source - if not,
            # this may/will give a different compilation result.
            global _module_map
            if len(_module_map) != len(sys.modules):  # pragma: no branch
                _module_map = {id(module): name for name, module in sys.modules.items()}
            imports = [
                (module_name, local_name)
                for local_name, module in referenced_modules
                if (module_name := _module_map.get(id(module))) is not None
            ]
            import_fragments = [f"{name} as {asname}" for name, asname in set(imports)]
            import_str = f"import {','.join(import_fragments)}\n"
        else:
            import_str = ""
        exec_str = (
            f"{import_str}def __construct_lambda(): {capture_str} return ({source})"
        )
        exec(exec_str, f_globals)
        compiled = f_globals["__construct_lambda"]()

    return compiled


def _lambda_code_matches_node(f, node):
    try:
        compiled = _mimic_lambda_from_node(f, node)
    except (NameError, SyntaxError):  # pragma: no cover # source is generated from ast
        return False
    if _function_key(f) == _function_key(compiled):
        return True
    # Try harder
    compiled.__code__ = _normalize_code(f, compiled)
    return _function_key(f) == _function_key(compiled)


def _check_unknown_perfectly_aligned_lambda(candidate):
    # This is a monkeypatch point for our self-tests, to make unknown
    # lambdas raise.
    pass


def _lambda_description(f, leeway=50, *, fail_if_confused_with_perfect_candidate=False):
    if hasattr(f, "__wrapped_target"):
        f = f.__wrapped_target

    # You might be wondering how a lambda can have a return-type annotation?
    # The answer is that we add this at runtime, in new_given_signature(),
    # and we do support strange choices as applying @given() to a lambda.
    sig = inspect.signature(f)
    assert sig.return_annotation in (Parameter.empty, None), sig

    # Using pytest-xdist on Python 3.13, there's an entry in the linecache for
    # file "<string>", which then returns nonsense to getsource.  Discard it.
    linecache.cache.pop("<string>", None)

    def format_lambda(body):
        # The signature is more informative than the corresponding ast.unparse
        # output in the case of default argument values, so add the signature
        # to the unparsed body
        return (
            f"lambda {str(sig)[1:-1]}: {body}" if sig.parameters else f"lambda: {body}"
        )

    if_confused = format_lambda("<unknown>")

    try:
        source_lines, lineno0 = inspect.findsource(f)
        source_lines = tuple(source_lines)  # make it hashable
    except OSError:
        return if_confused

    try:
        all_lambdas = AST_LAMBDAS_CACHE[source_lines]
    except KeyError:
        # The source isn't already parsed, so we try to shortcut by parsing just
        # the local block. If that fails to produce a code-identical lambda,
        # fall through to the full parse.
        local_lines = inspect.getblock(source_lines[lineno0:])
        local_block = textwrap.dedent("".join(local_lines))
        # The fairly common ".map(lambda x: ...)" case. This partial block
        # isn't valid syntax, but it might be if we remove the leading ".".
        local_block = local_block.removeprefix(".")

        try:
            local_tree = ast.parse(local_block)
        except SyntaxError:
            pass
        else:
            local_lambdas = extract_all_lambdas(local_tree)
            for candidate in local_lambdas:
                if reflection.ast_arguments_matches_signature(
                    candidate.args, sig
                ) and _lambda_code_matches_node(f, candidate):
                    return format_lambda(ast.unparse(candidate.body))

        # Local parse failed or didn't produce a match, go ahead with the full parse
        try:
            tree = ast.parse("".join(source_lines))
        except SyntaxError:
            all_lambdas = []
        else:
            all_lambdas = extract_all_lambdas(tree)
        AST_LAMBDAS_CACHE[source_lines] = all_lambdas

    aligned_lambdas = []
    for candidate in all_lambdas:
        if (
            candidate.lineno - leeway <= lineno0 + 1 <= candidate.lineno + leeway
            and reflection.ast_arguments_matches_signature(candidate.args, sig)
        ):
            aligned_lambdas.append(candidate)

    aligned_lambdas.sort(key=lambda c: abs(lineno0 + 1 - c.lineno))
    for candidate in aligned_lambdas:
        if _lambda_code_matches_node(f, candidate):
            return format_lambda(ast.unparse(candidate.body))

    # None of the aligned lambdas match perfectly in generated code.
    if aligned_lambdas and aligned_lambdas[0].lineno == lineno0 + 1:
        _check_unknown_perfectly_aligned_lambda(aligned_lambdas[0])

    return if_confused


def lambda_description(f):
    """
    Returns a syntactically-valid expression describing `f`. This is often, but
    not always, the exact lambda definition string which appears in the source code.
    The difference comes from parsing the lambda ast into `tree` and then returning
    the result of `ast.unparse(tree)`, which may differ in whitespace, double vs
    single quotes, etc.

    Returns a string indicating an unknown body if the parsing gets confused in any way.
    """
    try:
        return LAMBDA_DESCRIPTION_CACHE[f]
    except KeyError:
        pass

    key = _function_key(f, bounded_size=True)
    location = (f.__code__.co_filename, f.__code__.co_firstlineno)
    try:
        description, failed_locations = LAMBDA_DIGEST_DESCRIPTION_CACHE[key]
    except KeyError:
        failed_locations = set()
    else:
        # We got a hit in the digests cache, but only use it if either it has
        # a good (known) description, or if it is unknown but we already tried
        # to parse its exact source location before.
        if "<unknown>" not in description or location in failed_locations:
            # use the cached result
            LAMBDA_DESCRIPTION_CACHE[f] = description
            return description

    description = _lambda_description(f)
    LAMBDA_DESCRIPTION_CACHE[f] = description
    if "<unknown>" in description:
        failed_locations.add(location)
    else:
        failed_locations.clear()  # we have a good description now
    LAMBDA_DIGEST_DESCRIPTION_CACHE[key] = description, failed_locations
    return description
