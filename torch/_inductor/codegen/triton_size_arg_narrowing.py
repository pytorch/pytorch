"""Selective 32-bit size arguments for proven kernel classes.

`signature_to_meta` declares every `ks*` size argument `tl.int64`. Its reason is that a product of symbols may
overflow even when each symbol fits in int32. This module recognizes kernels for which a machine-checked proof
shows that declaring those arguments `i32` changes no observable behaviour. For such a kernel it returns the
argument names that may be narrowed; for every other kernel it returns nothing.

There is one supported class today: the pointwise kernel Inductor emits for a nested `cat` of six segments along
a dynamic last dimension (pytorch/pytorch#189940). Eligibility needs all of the following.

C1 The kernel matches the proved class exactly:
   - a pointwise heuristic, a 1-D grid, no fixed config;
   - the proved argument list and signature;
   - a body whose indexing, masks, payload dataflow and dead statements equal the proved IR, compared by a SHA-256
     of their canonical form.
C2 Inductor's symbolic state, not size hints, establishes:
   - H1' ks1..ks6 are size symbols with value-range lower bound >= 1;
   - H2  ks0 == ks1 + ... + ks6 (its precomputed definition);
   - H3' numel == s * ks0 for a size symbol s with lower bound >= 1;
   - H4  index dtype int32, and numel <= 2**31 - 1 guarded (or statically known).
         Dynamo re-checks the guard every call. When it fails, the recompiled kernel has xnumel: i64 and is not
         in the class.

The proof: NarrowCat.narrow_cat_equiv (Lean 4). Under C1-C2, for every lane of the 1-D grid, the i64 and i32
kernels agree on the store mask, the store address, the stored value, and every memory read. The companion
lemmas give equal, positive divisors (no new UB) and exact i32 packing of every size argument.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Any, cast

from torch.utils._ordered_set import OrderedSet


# SHA-256 of the canonical IR of the proved kernel (see _canonical).
_PROVED_CAT6_SHA256 = "34f3438bd120082a5d7313f7c2a72784893577c12e792e2fa9fdf1c9ef99576c"
_PROVED_ARGS = (
    "in_ptr0",
    "in_ptr1",
    "in_ptr2",
    "in_ptr3",
    "in_ptr4",
    "in_ptr5",
    "in_ptr6",
    "in_ptr7",
    "out_ptr0",
    "ks0",
    "ks1",
    "ks2",
    "ks3",
    "ks4",
    "ks5",
    "ks6",
    "xnumel",
    "XBLOCK",
)
_KS = tuple(f"ks{i}" for i in range(7))
_INT_MAX = 2**31 - 1
_PROLOGUE = (
    "xoffset = tl.program_id(0) * XBLOCK",
    "xindex = xoffset + tl.arange(0, XBLOCK)[:]",
    "xmask = xindex < xnumel",
)


class _Unsupported(Exception):
    pass


def _dtype(node: ast.AST) -> str:
    s = ast.unparse(node)
    if s in ("tl.int32", "tl.int64"):
        return s
    raise _Unsupported(s)


def _canonical(arg_names: list[str], body: str) -> tuple[Any, ...]:
    """Parse the kernel body into (store, dead statements). Raises _Unsupported on any unrecognized construct."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if tuple(ln.strip() for ln in lines[:3]) != _PROLOGUE:
        raise _Unsupported("prologue")
    ptrs = [a for a in arg_names if a.startswith(("in_ptr", "out_ptr"))]
    stmts = ast.parse("\n".join(ln.strip() for ln in lines[3:])).body
    env: dict[str, tuple[str, Any]] = {
        "xindex": ("int", ("xindex",)),
        "xmask": ("bool", ("xmask",)),
    }
    used: OrderedSet[str] = OrderedSet()

    def ref(name: str, kind: str) -> Any:
        if name in _KS:
            if kind != "int":
                raise _Unsupported(name)
            return ("ks", int(name[2:]))
        k, v = env.get(name, (None, None))
        if k != kind:
            raise _Unsupported(name)
        used.add(name)
        return v

    def ie(n: ast.AST) -> Any:
        if isinstance(n, ast.Name):
            return ref(n.id, "int")
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return ("lit", n.value)
        if (
            isinstance(n, ast.UnaryOp)
            and isinstance(n.op, ast.USub)
            and isinstance(n.operand, ast.Constant)
        ):
            return ("lit", -cast(Any, n.operand.value))
        if isinstance(n, ast.BinOp):
            ops: dict[type[ast.operator], str] = {
                ast.Add: "add",
                ast.Mult: "mul",
                ast.Mod: "tmod",
                ast.FloorDiv: "tdiv",
            }
            op = ops.get(type(n.op))
            if op:
                return (op, ie(n.left), ie(n.right))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            f = ast.unparse(n.func)
            if n.func.attr == "to" and len(n.args) == 1:
                return ("cast", ie(n.func.value), _dtype(n.args[0]))
            if f == "tl.broadcast_to" and ast.unparse(n.args[1]) == "[XBLOCK]":
                return ie(n.args[0])
            if (
                f == "tl.full"
                and ast.unparse(n.args[0]) == "[1]"
                and type(getattr(n.args[1], "value", None)) is int
            ):
                return ("slit", cast(ast.Constant, n.args[1]).value, _dtype(n.args[2]))
        raise _Unsupported(ast.unparse(n)[:60])

    def be(n: ast.AST) -> Any:
        if isinstance(n, ast.Name):
            return ref(n.id, "bool")
        if (
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and type(n.ops[0]) in (ast.Lt, ast.GtE)
        ):
            return (
                "lt" if isinstance(n.ops[0], ast.Lt) else "ge",
                ie(n.left),
                ie(n.comparators[0]),
            )
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
            return ("and", be(n.left), be(n.right))
        raise _Unsupported(ast.unparse(n)[:60])

    def fe(n: ast.AST) -> Any:
        if isinstance(n, ast.Name):
            return ref(n.id, "float")
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return ("fadd", fe(n.left), fe(n.right))
        if isinstance(n, ast.Call):
            f = ast.unparse(n.func)
            if f == "tl.where":
                return ("where", be(n.args[0]), fe(n.args[1]), fe(n.args[2]))
            if (
                f == "tl.full"
                and type(getattr(n.args[1], "value", None)) is float
                and cast(ast.Constant, n.args[1]).value == 0.0
                and ast.unparse(n.args[0]) != "[1]"
            ):
                return ("fzero",)
            if (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "to"
                and isinstance(n.func.value, ast.Call)
            ):
                ld = n.func.value
                if ast.unparse(ld.func) == "tl.load":
                    kw = {k.arg: ast.unparse(k.value) for k in ld.keywords}
                    a0 = ld.args[0]
                    if (
                        kw.get("other") == "0.0"
                        and OrderedSet(kw) <= OrderedSet(["other", "eviction_policy"])
                        and isinstance(a0, ast.BinOp)
                        and isinstance(a0.op, ast.Add)
                        and isinstance(a0.left, ast.Name)
                        and a0.left.id in ptrs
                    ):
                        return (
                            "load",
                            ptrs.index(a0.left.id),
                            ie(a0.right),
                            be(ld.args[1]),
                        )
        raise _Unsupported(ast.unparse(n)[:60])

    store = None
    for s in stmts:
        if (
            isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
        ):
            for kind, f in (("float", fe), ("bool", be), ("int", ie)):
                snapshot = OrderedSet(used)
                try:
                    env[s.targets[0].id] = (kind, f(s.value))
                    break
                except (_Unsupported, AttributeError, IndexError):
                    used.clear()
                    used.update(snapshot)
            else:
                raise _Unsupported(ast.unparse(s)[:60])
        elif (
            isinstance(s, ast.Expr)
            and isinstance(s.value, ast.Call)
            and ast.unparse(s.value.func) == "tl.store"
            and store is None
        ):
            a0 = s.value.args[0]
            if not (
                isinstance(a0, ast.BinOp)
                and isinstance(a0.left, ast.Name)
                and a0.left.id in ptrs
            ):
                raise _Unsupported("store")
            store = (
                ptrs.index(a0.left.id),
                ie(a0.right),
                fe(s.value.args[1]),
                be(s.value.args[2]),
            )
        else:
            raise _Unsupported(ast.unparse(s)[:60])
    if store is None:
        raise _Unsupported("no store")
    dead = tuple(
        sorted(
            (k, v[1])
            for k, v in env.items()
            if k not in used and k not in ("xindex", "xmask")
        )
    )
    return (tuple(arg_names), store, dead)


def canonical_sha256(arg_names: list[str], body: str) -> str:
    return hashlib.sha256(repr(_canonical(arg_names, body)).encode()).hexdigest()


def _symbolic_facts(kernel: Any, numel: Any) -> dict[str, Any]:
    import sympy

    from ..virtualized import V

    sv = V.graph.sizevars
    se = sv.shape_env
    name_of = {v: k for k, v in kernel.args.sizevars.items()}
    full: dict[str, Any] = {
        k: sv.inv_precomputed_replacements.get(name_of[k], name_of[k]) for k in _KS
    }

    def lower_ge1(e: Any) -> bool:
        rng = se.var_to_range.get(e) if isinstance(e, sympy.Symbol) else None
        return rng is not None and rng.lower >= 1

    facts: dict[str, Any] = {}
    facts["H1"] = all(lower_ge1(full[k]) for k in _KS[1:])
    facts["H2"] = sympy.expand(full["ks0"] - sum(full[k] for k in _KS[1:])) == 0
    numel = sympy.expand(numel)
    s = sympy.simplify(numel / full["ks0"])
    facts["H3"] = lower_ge1(s) and sympy.expand(s * full["ks0"] - numel) == 0
    guard = sympy.Le(numel, _INT_MAX)
    installed = str(guard) in {str(g.expr) for g in se.guards}
    facts["H4"] = str(kernel.index_dtype) == "tl.int32" and (
        installed or bool(sv.statically_known_true(guard))
    )
    return facts


def proven_int32_size_args(
    kernel: Any,
    arg_names: list[str],
    signature: dict[str, str],
    heuristic: str,
    grid_type: str,
    body: str,
) -> list[str]:
    """Names of size arguments that may be declared i32 for this kernel; [] unless C1 and C2 hold."""
    if (
        heuristic != "pointwise"
        or grid_type != "Grid1D"
        or kernel.fixed_config is not None
    ):
        return []
    if (
        kernel.inside_reduction
        or kernel.cooperative_reduction
        or tuple(arg_names) != _PROVED_ARGS
    ):
        return []
    if any(signature.get(k) != "i64" for k in _KS) or signature.get("xnumel") != "i32":
        return []
    if any(not signature.get(p, "").startswith("*") for p in _PROVED_ARGS[:9]):
        return []
    try:
        if canonical_sha256(arg_names, body) != _PROVED_CAT6_SHA256:
            return []
        facts = _symbolic_facts(kernel, kernel.numels["x"])
    except Exception:
        return []
    return list(_KS) if all(facts.values()) else []
