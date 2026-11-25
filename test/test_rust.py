# Owner(s): ["module: devx"]

import _thread
import contextlib
import dis
import gc
import importlib
import inspect
import io
import re
import signal
import sys
import traceback
import types
import unittest
import weakref

import torch
import torch.nn.functional as F
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
class TestRustBindings(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_function(self):
        self.assertEqual(torch._rust.test_function(), "hello from rust")

    def test_tensor_size(self):
        t = torch.zeros(2, 3, 4)
        self.assertEqual(torch._rust.tensor_size(t), [2, 3, 4])

    def test_tensor_size_scalar(self):
        t = torch.tensor(0.0)
        self.assertEqual(torch._rust.tensor_size(t), [])

    def test_tensor_size_1d(self):
        t = torch.zeros(5)
        self.assertEqual(torch._rust.tensor_size(t), [5])

    def test_collect_tensors_single(self):
        t = torch.zeros(3)
        result = torch._rust.collect_tensors(t)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], t)
        self.assertIsInstance(result[0], torch.Tensor)

    def test_collect_tensors_non_tensor(self):
        self.assertEqual(torch._rust.collect_tensors(42), [])
        self.assertEqual(torch._rust.collect_tensors("hello"), [])
        self.assertEqual(torch._rust.collect_tensors(None), [])
        self.assertEqual(torch._rust.collect_tensors(b"bytes"), [])

    def test_collect_tensors_list(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors([a, 1, b, "x"])
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_tuple(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors((a, b))
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_dict_collects_values(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({"a": a, "b": b})
        self.assertEqual([id(t) for t in result], [id(a), id(b)])

    def test_collect_tensors_dict_keys_ignored(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({a: b})
        self.assertEqual([id(t) for t in result], [id(b)])

    def test_collect_tensors_set(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors({a, b})
        self.assertEqual({id(t) for t in result}, {id(a), id(b)})

    def test_collect_tensors_frozenset(self):
        a, b = torch.zeros(2), torch.ones(3)
        result = torch._rust.collect_tensors(frozenset([a, b]))
        self.assertEqual({id(t) for t in result}, {id(a), id(b)})

    def test_collect_tensors_nested(self):
        a, b, c = torch.zeros(1), torch.ones(2), torch.zeros(3)
        result = torch._rust.collect_tensors([a, (b,), {"k": [c]}])
        self.assertEqual([id(t) for t in result], [id(a), id(b), id(c)])

    def test_collect_tensors_empty_containers(self):
        self.assertEqual(torch._rust.collect_tensors([]), [])
        self.assertEqual(torch._rust.collect_tensors(()), [])
        self.assertEqual(torch._rust.collect_tensors({}), [])
        self.assertEqual(torch._rust.collect_tensors(set()), [])


def _leaf():
    return 1


def _middle():
    return sum(_leaf() for _ in range(3))


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
@unittest.skipIf(sys.version_info < (3, 12), "requires Python 3.12+")
class TestCompile3(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def _run(self, fn, *args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = fn(*args)
        return result, out.getvalue()

    def test_prints_callees_in_order(self):
        @torch._compile3
        def top(x):
            return _middle() + x

        result, out = self._run(top, 10)
        self.assertEqual(result, 13)
        headers = [line for line in out.splitlines() if line.startswith("=== ")]
        names = [h.split()[1] for h in headers]
        self.assertEqual(
            names,
            [
                "TestCompile3.test_prints_callees_in_order.<locals>.top",
                "_middle",
                "_middle.<locals>.<genexpr>",
                "_leaf",
            ],
        )
        self.assertIn("RETURN_VALUE", out)
        self.assertIn("--- CFG ---", out)

    def test_each_code_printed_once_per_call(self):
        @torch._compile3
        def top():
            return [_leaf() for _ in range(5)]

        _, out = self._run(top)
        self.assertEqual(out.count("=== _leaf "), 1)
        _, out = self._run(top)
        self.assertEqual(out.count("=== _leaf "), 1)

    def test_disabled_after_return_and_raise(self):
        @torch._compile3
        def boom():
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            self._run(boom)
        _, out = self._run(_middle)
        self.assertEqual(out, "")

    def test_preserves_metadata(self):
        def documented():
            """doc"""

        wrapped = torch._compile3(documented)
        self.assertEqual(wrapped.__name__, "documented")
        self.assertEqual(wrapped.__doc__, "doc")


# Subjects for TestBytecodeToCfg, kept at module level so their reprs are short.
def _cfg_straight(a, b):
    c = a + b
    return c * 2


def _cfg_if(c, a):
    if c:
        x = a
    else:
        x = 1
    return x


def _cfg_while(n):
    i = 0
    while i < n:
        i += 1
    return i


def _cfg_for(xs):
    t = 0
    for x in xs:
        t += x
    return t


def _cfg_try_except(a):
    try:
        y = a.f()
        a.g(y)
    except ValueError:
        return y
    return 0


def _cfg_try_finally(a):
    try:
        a.f()
    finally:
        a.g()


def _cfg_with(m):
    with m:
        m.work()


def _cfg_nested_with(a, b):
    with a:
        with b:
            a.work(b)


def _cfg_gen(xs):
    for x in xs:  # noqa: UP028 (tests FOR_ITER + YIELD_VALUE)
        yield x


def _cfg_yield_from(it):
    r = yield from it
    return r


def _cfg_comprehension(xs):
    return [x + 1 for x in xs]


def _cfg_genexpr(xs):
    return sum(x for x in xs)


def _cfg_closure(x):
    y = 1
    return lambda: x + y


def _cfg_captured_comprehension(xs):
    return [lambda: x for x in xs]


def _cfg_maybe_del(c):
    if c:
        x = 1
    del x


def _cfg_with_then_use(m):
    with m:
        y = m.g()
    return y


def _cfg_only_is_none(x):
    try:
        x is None  # noqa: B015 (a try body that can't raise)
    except ValueError:
        pass


def _cfg_class_body():
    x = 1

    class Inner:
        y = x

    return Inner


# After `except ...: pass` the handler jumps to the code after the try, which
# nothing else reaches: a chain of two blocks. Raises in it unwind to the
# second try's handler, outside the chain.
def _cfg_outline(x):
    try:
        return x.a
    except AttributeError:
        pass
    try:
        return x.b
    except KeyError:
        return "key"


_CFG_MIN_ARG_OPS = {
    "SWAP",
    "COPY",
    "LIST_APPEND",
    "SET_ADD",
    "MAP_ADD",
    "LIST_EXTEND",
    "SET_UPDATE",
    "DICT_MERGE",
    "DICT_UPDATE",
}


def _nested_code(fn):
    return next(c for c in fn.__code__.co_consts if hasattr(c, "co_code"))


def _all_code_objects(co):
    yield co
    for c in co.co_consts:
        if hasattr(c, "co_code"):
            yield from _all_code_objects(c)


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
@unittest.skipIf(sys.version_info[:2] != (3, 12), "bytecode_to_cfg handles 3.12 only")
class TestBytecodeToCfg(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def cfg(self, fn, simplify=True, origins=False):
        return torch._rust.bytecode_to_cfg(fn, simplify=simplify, origins=origins)

    def test_opcode_table(self):
        real = {
            name: op
            for name, op in dis.opmap.items()
            if op < 256 and not name.startswith("INSTRUMENTED_")
        }
        self.assertEqual(dict(torch._rust._cfg_opcodes()), real)
        for name, op in real.items():
            if name in ("CACHE", "RESERVED", "INTERPRETER_EXIT"):
                continue
            has_arg = op >= dis.HAVE_ARGUMENT
            for arg in (0, 1, 2, 3, 4, 7, 15, 0x102) if has_arg else (0,):
                for jump in (False, True):
                    try:
                        want = dis.stack_effect(op, arg if has_arg else None, jump=jump)
                    except ValueError:
                        continue
                    sig = torch._rust._cfg_signature(op, arg, jump)
                    # Opargs the compiler never emits (e.g. SWAP 1, LIST_APPEND 0).
                    if sig is None and arg < 2 and name in _CFG_MIN_ARG_OPS:
                        continue
                    self.assertIsNotNone(sig, f"{name} {arg}")
                    got = len(sig[1]) - len(sig[0])
                    if name == "FOR_ITER" and jump:
                        # dis models the jump to END_FOR; we jump past it.
                        want -= 2
                    if name == "RETURN_GENERATOR":
                        # We push the value that the following POP_TOP drops.
                        want += 1
                    self.assertEqual(got, want, f"{name} arg={arg} jump={jump}")

    def test_signatures(self):
        def sig(name, arg=0):
            return torch._rust._cfg_signature(dis.opmap[name], arg, False)

        self.assertEqual(sig("CHECK_EXC_MATCH"), ([True, True], [0, -1]))
        self.assertEqual(sig("CHECK_EG_MATCH"), ([True, True], [-1, -1]))
        self.assertEqual(sig("END_ASYNC_FOR"), ([True, True], []))
        self.assertEqual(sig("MATCH_CLASS", 1), ([True, True, True], [-1]))
        self.assertEqual(sig("MATCH_KEYS"), ([True, True], [0, 1, -1]))
        self.assertEqual(sig("GET_LEN"), ([True], [0, -1]))
        self.assertEqual(sig("GET_ANEXT"), ([True], [0, -1]))
        self.assertEqual(
            sig("WITH_EXCEPT_START"), ([True, False, False, True], [0, 1, 2, 3, -1])
        )
        self.assertEqual(sig("LIST_APPEND", 2), ([True, False, True], [0, 1]))
        self.assertEqual(sig("PUSH_EXC_INFO"), ([True], [-1, 0]))

    def test_straight(self):
        self.assertExpectedInline(
            self.cfg(_cfg_straight),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = LoadLocal b
  %2 = BinaryOp + %0, %1
  StoreLocal c %2
  %3 = LoadLocal c
  %4 = Const 2
  %5 = BinaryOp * %3, %4
  return %5
""",
        )

    def test_if(self):
        self.assertExpectedInline(
            self.cfg(_cfg_if),
            """\
bb0:
  Resume
  %0 = LoadLocal c
  %1 = ToBool %0
  branch %1, bb1, bb2
bb1:
  %2 = LoadLocal a
  StoreLocal x %2
  %3 = LoadLocal x
  return %3
bb2:
  %4 = Const 1
  StoreLocal x %4
  %5 = LoadLocal x
  return %5
""",
        )

    def test_while(self):
        self.assertExpectedInline(
            self.cfg(_cfg_while),
            """\
bb0:
  Resume
  %0 = Const 0
  StoreLocal i %0
  %1 = LoadLocal i
  %2 = LoadLocal n
  %3 = CompareOp < %1, %2
  %4 = ToBool %3
  branch %4, bb1, bb3
bb1:
  %5 = LoadLocal i
  %6 = Const 1
  %7 = BinaryOp += %5, %6
  StoreLocal i %7
  %8 = LoadLocal i
  %9 = LoadLocal n
  %10 = CompareOp < %8, %9
  %11 = ToBool %10
  branch %11, bb2, bb3
bb2:
  EvalBreaker
  jump bb1
bb3:
  %12 = LoadLocal i
  return %12
""",
        )

    def test_for(self):
        self.assertExpectedInline(
            self.cfg(_cfg_for),
            """\
bb0:
  Resume
  %0 = Const 0
  StoreLocal t %0
  %1 = LoadLocal xs
  %2 = GetIter %1
  jump bb1(s0=%2)
bb1(%3 s0):
  %4 = IterNext %3
  %5 = Project 0 %4
  %6 = Project 1 %4
  branch %6, bb2(s0=%3, s1=%5), bb3
bb2(%7 s0, %8 s1):
  StoreLocal x %8
  %9 = LoadLocal t
  %10 = LoadLocal x
  %11 = BinaryOp += %9, %10
  StoreLocal t %11
  EvalBreaker
  jump bb1(s0=%7)
bb3:
  %12 = LoadLocal t
  return %12
""",
        )

    def test_for_unsimplified(self):
        self.assertExpectedInline(
            self.cfg(_cfg_for, simplify=False),
            """\
bb0:
  Resume
  %0 = Const 0
  StoreLocal t %0
  %1 = LoadLocal xs
  %2 = GetIter %1
  jump bb1(s0=%2)
bb1(%3 s0):
  %4 = IterNext %3
  %5 = Project 0 %4
  %6 = Project 1 %4
  branch %6, bb2(s0=%3, s1=%5), bb3
bb2(%7 s0, %8 s1):
  StoreLocal x %8
  %9 = LoadLocal t
  %10 = LoadLocal x
  %11 = BinaryOp += %9, %10
  StoreLocal t %11
  EvalBreaker
  jump bb1(s0=%7)
bb3:
  %12 = LoadLocal t
  return %12
""",
        )

    def test_try_except(self):
        # One block for the try body, with two unwind segments.
        self.assertExpectedInline(
            self.cfg(_cfg_try_except),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = LoadMethod f %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = Call 0 %2, %3
  StoreLocal y %4
  %5 = LoadLocal a
  %6 = LoadMethod g %5
  %7 = Project 0 %6
  %8 = Project 1 %6
  %9 = LoadLocal y
  %10 = Call 1 %7, %8, %9
  %11 = Const 0
  return %11
  unwind [2..14) -> bb1
bb1(%12 exc):
  %13 = PUSH_EXC_INFO 0 %12
  %14 = LoadGlobal ValueError
  %15 = CHECK_EXC_MATCH 0 %12, %14
  %16 = ToBool %15
  branch %16, bb2(s0=%13), bb3(s0=%13, s1=%12)
  unwind [1..4) -> bb4(s0=%13)
bb2(%17 s0):
  %18 = LoadLocalChecked y
  POP_EXCEPT 0 %17
  return %18
  unwind [0..2) -> bb4(s0=%17)
bb3(%19 s0, %20 s1):
  reraise %20 -> bb4(s0=%19)
bb4(%21 s0, %22 lasti, %23 exc):
  POP_EXCEPT 0 %21
  reraise %23, lasti=%22
""",
        )

    def test_try_finally(self):
        self.assertExpectedInline(
            self.cfg(_cfg_try_finally),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = LoadMethod f %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = Call 0 %2, %3
  %5 = LoadLocal a
  %6 = LoadMethod g %5
  %7 = Project 0 %6
  %8 = Project 1 %6
  %9 = Call 0 %7, %8
  %10 = Const None
  return %10
  unwind [2..7) -> bb1
  unwind [7..12) -> raise
bb1(%11 exc):
  %12 = PUSH_EXC_INFO 0 %11
  %13 = LoadLocal a
  %14 = LoadMethod g %13
  %15 = Project 0 %14
  %16 = Project 1 %14
  %17 = Call 0 %15, %16
  reraise %11 -> bb2(s0=%12)
  unwind [2..6) -> bb2(s0=%12)
bb2(%18 s0, %19 lasti, %20 exc):
  POP_EXCEPT 0 %18
  reraise %20, lasti=%19
""",
        )

    def test_with(self):
        self.assertExpectedInline(
            self.cfg(_cfg_with),
            """\
bb0:
  Resume
  %0 = LoadLocal m
  %1 = BEFORE_WITH 0 %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = LoadLocal m
  %5 = LoadMethod work %4
  %6 = Project 0 %5
  %7 = Project 1 %5
  %8 = Call 0 %6, %7
  %9 = Const None
  %10 = Const None
  %11 = Const None
  %12 = Call 2 %2, %9, %10, %11
  %13 = Const None
  return %13
  unwind [6..13) -> bb1(s0=%2)
  unwind [13..15) -> raise
bb1(%14 s0, %15 lasti, %16 exc):
  %17 = PUSH_EXC_INFO 0 %16
  %18 = WITH_EXCEPT_START 0 %14, %16
  %19 = ToBool %18
  branch %19, bb2(s2=%17), bb3(s1=%15, s2=%17, s3=%16)
  unwind [1..3) -> bb4(s2=%17)
bb2(%20 s2):
  POP_EXCEPT 0 %20
  %21 = Const None
  return %21
bb3(%22 s1, %23 s2, %24 s3):
  reraise %24, lasti=%22 -> bb4(s2=%23)
bb4(%25 s2, %26 lasti, %27 exc):
  POP_EXCEPT 0 %25
  reraise %27, lasti=%26
""",
        )

    def test_nested_with(self):
        self.assertExpectedInline(
            self.cfg(_cfg_nested_with),
            """\
bb0:
  Resume
  %0 = LoadLocal a
  %1 = BEFORE_WITH 0 %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = LoadLocal b
  %5 = BEFORE_WITH 0 %4
  %6 = Project 0 %5
  %7 = Project 1 %5
  %8 = LoadLocal a
  %9 = LoadMethod work %8
  %10 = Project 0 %9
  %11 = Project 1 %9
  %12 = LoadLocal b
  %13 = Call 1 %10, %11, %12
  %14 = Const None
  %15 = Const None
  %16 = Const None
  %17 = Call 2 %6, %14, %15, %16
  jump bb3(s0=%2)
  unwind [6..10) -> bb6(s0=%2)
  unwind [10..18) -> bb1(s0=%2, s1=%6)
  unwind [18..19) -> bb6(s0=%2)
bb1(%18 s0, %19 s1, %20 lasti, %21 exc):
  %22 = PUSH_EXC_INFO 0 %21
  %23 = WITH_EXCEPT_START 0 %19, %21
  %24 = ToBool %23
  branch %24, bb2(s0=%18, s3=%22), bb4(s0=%18, s2=%20, s3=%22, s4=%21)
  unwind [1..3) -> bb5(s0=%18, s3=%22)
bb2(%25 s0, %26 s3):
  POP_EXCEPT 0 %26
  EvalBreaker
  jump bb3(s0=%25)
  unwind [1..2) -> bb6(s0=%25)
bb3(%27 s0):
  %28 = Const None
  %29 = Const None
  %30 = Const None
  %31 = Call 2 %27, %28, %29, %30
  %32 = Const None
  return %32
bb4(%33 s0, %34 s2, %35 s3, %36 s4):
  reraise %36, lasti=%34 -> bb5(s0=%33, s3=%35)
bb5(%37 s0, %38 s3, %39 lasti, %40 exc):
  POP_EXCEPT 0 %38
  reraise %40, lasti=%39 -> bb6(s0=%37)
bb6(%41 s0, %42 lasti, %43 exc):
  %44 = PUSH_EXC_INFO 0 %43
  %45 = WITH_EXCEPT_START 0 %41, %43
  %46 = ToBool %45
  branch %46, bb7(s2=%44), bb8(s1=%42, s2=%44, s3=%43)
  unwind [1..3) -> bb9(s2=%44)
bb7(%47 s2):
  POP_EXCEPT 0 %47
  %48 = Const None
  return %48
bb8(%49 s1, %50 s2, %51 s3):
  reraise %51, lasti=%49 -> bb9(s2=%50)
bb9(%52 s2, %53 lasti, %54 exc):
  POP_EXCEPT 0 %52
  reraise %54, lasti=%53
""",
        )

    def test_generator(self):
        self.assertExpectedInline(
            self.cfg(_cfg_gen),
            """\
bb0:
  %0 = RETURN_GENERATOR 0
  Resume
  %1 = LoadLocal xs
  %2 = GetIter %1
  jump bb1(s0=%2)
  unwind [1..4) -> bb4
bb1(%3 s0):
  %4 = IterNext %3
  %5 = Project 0 %4
  %6 = Project 1 %4
  branch %6, bb2(s0=%3, s1=%5), bb3
  unwind [0..3) -> bb4
bb2(%7 s0, %8 s1):
  StoreLocal x %8
  %9 = LoadLocal x
  %10 = Yield %7, %9
  Resume
  EvalBreaker
  jump bb1(s0=%7)
  unwind [2..5) -> bb4
bb3:
  %11 = Const None
  return %11
bb4(%12 lasti, %13 exc):
  %14 = CALL_INTRINSIC_1 3 %13
  reraise %14, lasti=%12
""",
        )

    def test_yield_from(self):
        self.assertExpectedInline(
            self.cfg(_cfg_yield_from),
            """\
bb0:
  %0 = RETURN_GENERATOR 0
  Resume
  %1 = LoadLocal it
  %2 = GET_YIELD_FROM_ITER 0 %1
  %3 = Const None
  jump bb1(s0=%2, s1=%3)
  unwind [1..5) -> bb5
bb1(%4 s0, %5 s1):
  %6 = Send %4, %5
  %7 = Project 0 %6
  %8 = Project 1 %6
  branch %8, bb2(s0=%4, s1=%7), bb4(s1=%7)
  unwind [0..3) -> bb5
bb2(%9 s0, %10 s1):
  %11 = Yield %9, %10
  Resume
  jump bb1(s0=%9, s1=%11)
  unwind [0..1) -> bb3(s0=%9)
  unwind [1..2) -> bb5
bb3(%12 s0, %13 exc):
  %14 = CLEANUP_THROW 0 %12, unknown, %13
  %15 = Project 0 %14
  %16 = Project 1 %14
  EvalBreaker
  jump bb4(s1=%16)
  unwind [0..3) -> bb5
  unwind [3..4) -> raise
bb4(%17 s1):
  StoreLocal r %17
  %18 = LoadLocal r
  return %18
bb5(%19 lasti, %20 exc):
  %21 = CALL_INTRINSIC_1 3 %20
  reraise %21, lasti=%19
""",
        )

    def test_comprehension(self):
        self.assertExpectedInline(
            self.cfg(_cfg_comprehension),
            """\
bb0:
  Resume
  %0 = LoadLocal xs
  %1 = GetIter %0
  %2 = LoadLocal x
  StoreLocal x null
  %3 = BUILD_LIST 0
  jump bb1(s0=%2, s1=%3, s2=%1)
bb1(%4 s0, %5 s1, %6 s2):
  %7 = IterNext %6
  %8 = Project 0 %7
  %9 = Project 1 %7
  branch %9, bb2(s0=%4, s1=%5, s2=%6, s3=%8), bb3(s0=%4, s1=%5)
  unwind [0..3) -> bb4(s0=%4)
bb2(%10 s0, %11 s1, %12 s2, %13 s3):
  StoreLocal x %13
  %14 = LoadLocal x
  %15 = Const 1
  %16 = BinaryOp + %14, %15
  LIST_APPEND 2 %11, %16
  EvalBreaker
  jump bb1(s0=%10, s1=%11, s2=%12)
  unwind [3..6) -> bb4(s0=%10)
bb3(%17 s0, %18 s1):
  StoreLocal x %17
  return %18
bb4(%19 s0, %20 exc):
  StoreLocal x %19
  reraise %20
""",
        )

    def test_genexpr(self):
        self.assertExpectedInline(
            self.cfg(_nested_code(_cfg_genexpr)),
            """\
bb0:
  %0 = RETURN_GENERATOR 0
  Resume
  %1 = LoadLocal .0
  jump bb1(s0=%1)
  unwind [1..3) -> bb4
bb1(%2 s0):
  %3 = IterNext %2
  %4 = Project 0 %3
  %5 = Project 1 %3
  branch %5, bb2(s0=%2, s1=%4), bb3
  unwind [0..3) -> bb4
bb2(%6 s0, %7 s1):
  StoreLocal x %7
  %8 = LoadLocal x
  %9 = Yield %6, %8
  Resume
  EvalBreaker
  jump bb1(s0=%6)
  unwind [2..5) -> bb4
bb3:
  %10 = Const None
  return %10
bb4(%11 lasti, %12 exc):
  %13 = CALL_INTRINSIC_1 3 %12
  reraise %13, lasti=%11
""",
        )

    def test_closure(self):
        self.assertExpectedInline(
            self.cfg(_cfg_closure),
            """\
bb0:
  MakeCell x
  MakeCell y
  Resume
  %0 = Const 1
  StoreDeref y %0
  %1 = LoadLocal x
  %2 = LoadLocal y
  %3 = BuildTuple 2 %1, %2
  %4 = Const <code _cfg_closure.<locals>.<lambda>>
  %5 = MakeFunction 0x8 %3, %4
  return %5
""",
        )
        self.assertExpectedInline(
            self.cfg(_nested_code(_cfg_closure)),
            """\
bb0:
  CopyFreeVars 2
  Resume
  %0 = LoadDeref x
  %1 = LoadDeref y
  %2 = BinaryOp + %0, %1
  return %2
""",
        )

    def test_captured_comprehension(self):
        self.assertExpectedInline(
            self.cfg(_cfg_captured_comprehension),
            """\
bb0:
  MakeCell x
  Resume
  %0 = LoadLocal xs
  %1 = GetIter %0
  %2 = LoadLocal x
  StoreLocal x null
  MakeCell x
  %3 = BUILD_LIST 0
  jump bb1(s0=%2, s1=%3, s2=%1)
bb1(%4 s0, %5 s1, %6 s2):
  %7 = IterNext %6
  %8 = Project 0 %7
  %9 = Project 1 %7
  branch %9, bb2(s0=%4, s1=%5, s2=%6, s3=%8), bb3(s0=%4, s1=%5)
  unwind [0..3) -> bb4(s0=%4)
bb2(%10 s0, %11 s1, %12 s2, %13 s3):
  StoreDeref x %13
  %14 = LoadLocal x
  %15 = BuildTuple 1 %14
  %16 = Const <code _cfg_captured_comprehension.<locals>.<lambda>>
  %17 = MakeFunction 0x8 %15, %16
  LIST_APPEND 2 %11, %17
  EvalBreaker
  jump bb1(s0=%10, s1=%11, s2=%12)
  unwind [6..7) -> bb4(s0=%10)
bb3(%18 s0, %19 s1):
  StoreLocal x %18
  return %19
bb4(%20 s0, %21 exc):
  StoreLocal x %20
  reraise %21
""",
        )

    def test_del_local(self):
        self.assertExpectedInline(
            self.cfg(_cfg_maybe_del),
            """\
bb0:
  Resume
  %0 = LoadLocal c
  %1 = ToBool %0
  branch %1, bb1, bb2
bb1:
  %2 = Const 1
  StoreLocal x %2
  jump bb2
bb2:
  DeleteLocal x
  %3 = Const None
  return %3
""",
        )

    def test_with_then_use(self):
        self.assertExpectedInline(
            self.cfg(_cfg_with_then_use),
            """\
bb0:
  Resume
  %0 = LoadLocal m
  %1 = BEFORE_WITH 0 %0
  %2 = Project 0 %1
  %3 = Project 1 %1
  %4 = LoadLocal m
  %5 = LoadMethod g %4
  %6 = Project 0 %5
  %7 = Project 1 %5
  %8 = Call 0 %6, %7
  StoreLocal y %8
  %9 = Const None
  %10 = Const None
  %11 = Const None
  %12 = Call 2 %2, %9, %10, %11
  %13 = LoadLocal y
  return %13
  unwind [6..14) -> bb1(s0=%2)
  unwind [14..16) -> raise
bb1(%14 s0, %15 lasti, %16 exc):
  %17 = PUSH_EXC_INFO 0 %16
  %18 = WITH_EXCEPT_START 0 %14, %16
  %19 = ToBool %18
  branch %19, bb2(s2=%17), bb3(s1=%15, s2=%17, s3=%16)
  unwind [1..3) -> bb4(s2=%17)
bb2(%20 s2):
  POP_EXCEPT 0 %20
  %21 = LoadLocalChecked y
  return %21
bb3(%22 s1, %23 s2, %24 s3):
  reraise %24, lasti=%22 -> bb4(s2=%23)
bb4(%25 s2, %26 lasti, %27 exc):
  POP_EXCEPT 0 %25
  reraise %27, lasti=%26
""",
        )

    def test_unreached_handler_dropped(self):
        self.assertExpectedInline(
            self.cfg(_cfg_only_is_none),
            """\
bb0:
  Resume
  %0 = LoadLocal x
  %1 = Const None
  %2 = IS_OP 0 %0, %1
  %3 = Const None
  return %3
""",
        )

    def test_origins(self):
        # Raising ops print the byte offset of the bytecode they came from.
        raising = {
            i.offset
            for i in dis.get_instructions(_cfg_try_except)
            if i.opname in ("LOAD_ATTR", "CALL", "LOAD_GLOBAL", "LOAD_FAST_CHECK")
        }
        out = self.cfg(_cfg_try_except, origins=True)
        printed = {int(m) for m in re.findall(r" @(\d+)", out)}
        self.assertTrue(raising <= printed, (raising, printed))
        self.assertExpectedInline(out.splitlines()[2], """  %0 = LoadLocal a""")

    def test_class_body(self):
        self.assertExpectedInline(
            self.cfg(_nested_code(_cfg_class_body)),
            """\
bb0:
  CopyFreeVars 1
  Resume
  %0 = LOAD_NAME 0
  STORE_NAME 1 %0
  %1 = Const '_cfg_class_body.<locals>.Inner'
  STORE_NAME 2 %1
  %2 = LOAD_LOCALS 0
  %3 = LoadFromDictOrDeref x %2
  STORE_NAME 3 %3
  %4 = Const None
  return %4
""",
        )

    def test_extended_arg(self):
        body = "".join(f"        x = x + {i}\n" for i in range(150))
        ns = {}
        exec(f"def big(x):\n    if x:\n{body}    return x\n", ns)
        big = ns["big"]
        self.assertTrue(
            any(i.opname == "EXTENDED_ARG" for i in dis.get_instructions(big))
        )
        out = self.cfg(big)
        self.assertIn("branch %1, bb1, bb2\n", out)
        self.assertTrue(
            out.endswith("  jump bb2\nbb2:\n  %452 = LoadLocal x\n  return %452\n")
        )

    def test_outline(self):
        self.assertExpectedInline(
            torch._rust.bytecode_to_cfg(_cfg_outline, outline="merge"),
            """\
bb0:
  Resume
  %0 = LoadLocal x
  %1 = LoadAttr a %0
  return %1
  unwind [2..3) -> bb1
bb1(%2 exc):
  %3 = PUSH_EXC_INFO 0 %2
  %4 = LoadGlobal AttributeError
  %5 = CHECK_EXC_MATCH 0 %2, %4
  %6 = ToBool %5
  branch %6, bb2(s0=%3), bb7(s0=%3, s1=%2)
  unwind [1..4) -> bb8(s0=%3)
bb2(%7 s0):
  POP_EXCEPT 0 %7
  %8 = LoadLocal x
  %9 = LoadAttr b %8
  return %9
  unwind [2..3) -> bb3
  origin:
    bb0(%0 s0):
      POP_EXCEPT 0 %0
      jump bb1
    bb1:
      %1 = LoadLocal x
      %2 = LoadAttr b %1
      leave(%2)
      unwind [1..2) -> outer 0()
bb3(%10 exc):
  %11 = PUSH_EXC_INFO 0 %10
  %12 = LoadGlobal KeyError
  %13 = CHECK_EXC_MATCH 0 %10, %12
  %14 = ToBool %13
  branch %14, bb4(s0=%11), bb5(s0=%11, s1=%10)
  unwind [1..4) -> bb6(s0=%11)
bb4(%15 s0):
  POP_EXCEPT 0 %15
  %16 = Const 'key'
  return %16
bb5(%17 s0, %18 s1):
  reraise %18 -> bb6(s0=%17)
bb6(%19 s0, %20 lasti, %21 exc):
  POP_EXCEPT 0 %19
  reraise %21, lasti=%20
bb7(%22 s0, %23 s1):
  reraise %23 -> bb8(s0=%22)
bb8(%24 s0, %25 lasti, %26 exc):
  POP_EXCEPT 0 %24
  reraise %26, lasti=%25
""",
        )

    def test_robustness_sweep(self):
        modules = [
            "json",
            "inspect",
            "dis",
            "ast",
            "asyncio.base_events",
            "contextlib",
            "typing",
            "torch.nn.modules.module",
        ]
        for name in modules:
            src = inspect.getsource(importlib.import_module(name))
            for co in _all_code_objects(compile(src, name, "exec")):
                # Raises NotImplementedError/RuntimeError on failure.
                torch._rust.bytecode_to_cfg(co, simplify=False)
                for outline in (None, "merge", "origin"):
                    torch._rust.bytecode_to_cfg(co, outline=outline)


def _exec_tb(e):
    out = []
    tb = e.__traceback__
    while tb is not None:
        out.append(
            (
                tb.tb_frame.f_code.co_name,
                tb.tb_lineno - tb.tb_frame.f_code.co_firstlineno,
            )
        )
        tb = tb.tb_next
    return out


# Here the chains' raises unwind to handlers outside them with stack values
# as args.
def _cfg_outline_context():
    context = Exception("context")
    try:
        raise context
    except Exception:
        try:
            raise Exception("caught")  # noqa: TRY002
        except Exception:
            pass
        try:
            raise Exception("new")  # noqa: TRY002
        except Exception as exc:
            raised = exc
    return raised.__context__ is context


def _cfg_outline_tb(fn, x):
    try:
        fn(x)
    except TypeError as e:
        return _exec_tb(e)


class _Del:
    log = []

    def __init__(self, n):
        self.n = n

    def __del__(self):
        _Del.log.append(self.n)


@unittest.skipIf(torch._rust is None, "torch._rust extension is not built")
@unittest.skipIf(sys.version_info[:2] != (3, 12), "the CFG executor handles 3.12 only")
class TestCfgExecution(TestCase):
    hw_classification = HardwareClassification.GENERIC

    # Runs fn normally and under the CFG executor, recording side effects and
    # types or not; results must match.
    def same(self, fn, *args):
        expected = fn(*args)
        self.assertEqual(torch._rust.call_with_cfg_execution(fn, *args), expected)
        torch._rust._set_cfg_record(False)
        try:
            self.assertEqual(torch._rust.call_with_cfg_execution(fn, *args), expected)
        finally:
            torch._rust._set_cfg_record(True)
        return expected

    def test_control_flow(self):
        def f(n):
            t = 0
            for i in range(n):
                if i % 3 == 0:
                    continue
                t += i
            while t > 10:
                t -= 7
            a, *b, c = [1, 2, 3, 4]
            return t, a, b, c, {"x": 1, "y": [i for i in range(3)]}, f"{t!r:>4}"  # noqa: C416

        self.same(f, 20)

    def test_exceptions_and_tracebacks(self):
        def deep():
            raise ValueError("boom")

        def f():
            try:
                deep()
            except ValueError as e:
                tb = _exec_tb(e)
            try:
                try:
                    1 / 0
                except ZeroDivisionError as e:
                    raise KeyError("k") from e
            except KeyError as e:
                chained = repr(e.__cause__)
            finally:
                done = sys.exception()
            return tb, chained, done

        self.assertEqual(self.same(f)[0], [("f", 2), ("deep", 1)])

    def test_unbound_local_and_name_error(self):
        def f():
            if False:
                y = 1
            try:
                return y
            except UnboundLocalError as e:
                msg = str(e)
            try:
                return undefined_name
            except NameError as e:
                return msg, str(e), e.name

        self.same(f)

    def test_generators_and_coroutines(self):
        def f():
            def inner():
                r = yield 1
                return r

            def outer():
                v = yield from inner()
                yield v

            g = outer()
            out = [next(g), g.send("sent")]
            gen = (x * x for x in range(4))

            async def coro(x):
                await asyncio.sleep(0)
                return x * 2

            async def agen():
                yield 1
                yield 2

            async def main():
                return await coro(21), [x async for x in agen()]

            return out, list(gen), asyncio.run(main())

        import asyncio

        self.same(f)

    def test_lifetimes(self):
        def f():
            _Del.log.clear()
            _Del(1)
            _Del.log.append("temp dropped")
            x = _Del(2)
            _Del.log.append("local alive")
            x = None  # noqa: F841
            _Del.log.append("local rebound")
            y = _Del(3)  # noqa: F841
            return list(_Del.log)

        self.same(f)

    def test_load_global_cache(self):
        # A LoadGlobal reuses its last lookup while the globals and builtins
        # dicts are unchanged.
        bt = {"len": "builtin len"}
        ns = {"__builtins__": bt}
        exec("def f():\n    return x, len\n", ns)
        f = ns["f"]

        def run():
            bt["len"] = "builtin len"
            out = []
            for x in (1, 2):
                ns["x"] = x
                out.append(f())
            ns["len"] = "global len"
            out.append(f())
            del ns["len"]
            bt["len"] = "new builtin len"
            out.append(f())
            out.append(types.FunctionType(f.__code__, {"__builtins__": bt, "x": 3})())
            del ns["x"]
            try:
                f()
            except NameError as e:
                out.append(str(e))
            return out

        self.assertEqual(
            self.same(run),
            [
                (1, "builtin len"),
                (2, "builtin len"),
                (2, "global len"),
                (2, "new builtin len"),
                (3, "new builtin len"),
                "name 'x' is not defined",
            ],
        )

    def test_recursion_limit(self):
        def rec(n):
            return rec(n + 1)

        def f():
            try:
                rec(0)
            except RecursionError as e:
                return str(e)

        self.same(f)

    def test_pep695_intrinsics(self):
        ns = {}
        src = "def f[T, *Ts, **P](x: T) -> T:\n    return x\ntype Alias[K] = list[K]\n"
        exec(src, ns)

        def g():
            p = ns["f"].__type_params__
            return [(type(t).__name__, t.__name__) for t in p], ns["Alias"].__value__

        self.same(g)

    def test_eval_breaker_on_generator_resume(self):
        class Boom(Exception):
            pass

        def handler(*args):
            raise Boom

        def f():
            def g():
                yield 0
                try:
                    yield 1
                except Boom:
                    yield "caught"

            it = g()
            next(it)
            next(it)
            # zip calls interrupt_main (which only trips the signal) and then
            # resumes g from C, so g's RESUME is the first eval-breaker check.
            return list(zip(map(_thread.interrupt_main, [signal.SIGUSR1]), it))

        old = signal.signal(signal.SIGUSR1, handler)
        try:
            self.assertEqual(self.same(f), [(None, "caught")])
        finally:
            signal.signal(signal.SIGUSR1, old)

    def test_outlined_execution(self):
        class A:
            a = 1

        class B:
            b = 2

        class K:
            @property
            def b(self):
                raise KeyError

        class T:
            @property
            def b(self):
                raise TypeError("t")

        # Fresh code objects: the executor caches each one's CFG.
        def fresh(f):
            return type(f)(f.__code__.replace(co_name=f.__name__), f.__globals__)

        for mode in ("merge", "origin"):
            fn = fresh(_cfg_outline)
            torch._rust._set_cfg_outline(mode)
            try:
                for x in (A(), B(), K()):
                    self.same(fn, x)
                self.same(_cfg_outline_tb, fn, T())
                self.assertTrue(self.same(fresh(_cfg_outline_context)))
            finally:
                torch._rust._set_cfg_outline(None)

    def test_print_torch_ops(self):
        def inner(t):
            return t.relu()

        def f(x, w):
            y = torch.nn.functional.linear(x, w)
            z = inner(y) + 1
            return z.sum()

        x = torch.ones(2, 3, requires_grad=True)
        w = torch.ones(4, 3)
        out = io.StringIO()
        torch._rust.set_print_torch_ops(True)
        try:
            with contextlib.redirect_stdout(out):
                self.assertEqual(self.same(f, x, w), f(x, w))
        finally:
            torch._rust.set_print_torch_ops(False)
        first = {"f": f.__code__.co_firstlineno, "inner": inner.__code__.co_firstlineno}
        # Lines relative to each function's def, so the golden survives edits.
        text = re.sub(
            r"# \S+\.(\w+) \(test_rust\.py:(\d+)\) @(\d+)",
            lambda m: f"# {m[1]} line +{int(m[2]) - first[m[1]]} @{m[3]}",
            out.getvalue(),
        )
        # Printed by both executor runs (recording and not).
        self.assertExpectedInline(
            text,
            """\
torch op: aten::linear(f32[2, 3], f32[4, 3], None)  # f line +1 @78
torch op: aten::relu(f32[2, 4])  # inner line +1 @24
torch op: aten::add.Tensor(f32[2, 4], i64[], alpha=1)  # f line +2 @104
torch op: aten::sum(f32[2, 4], dtype=None)  # f line +3 @132
torch op: aten::linear(f32[2, 3], f32[4, 3], None)  # f line +1 @78
torch op: aten::relu(f32[2, 4])  # inner line +1 @24
torch op: aten::add.Tensor(f32[2, 4], i64[], alpha=1)  # f line +2 @104
torch op: aten::sum(f32[2, 4], dtype=None)  # f line +3 @132
""",
        )
        # Off again: nothing is printed.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            torch._rust.call_with_cfg_execution(f, x, w)
        self.assertEqual(out.getvalue(), "")

    def test_print_torch_ops_with_hop(self):
        # Higher-order ops dispatch in Python and see PreDispatch enabled with
        # no pre-dispatch mode; they must skip it rather than fail.
        def f(x):
            return torch.cond(x.sum() > 0, torch.sin, torch.cos, (x,))

        x = torch.ones(3)
        torch._rust.set_print_torch_ops(True)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.same(f, x)
        finally:
            torch._rust.set_print_torch_ops(False)
        self.assertIn("aten::sin", out.getvalue())

    def test_side_effects(self):
        def py():
            return 1

        def via_c():
            return sorted([2, 1], key=lambda v: -v)

        def gen():
            yield torch.ones(2)

        def f(x):
            py()
            via_c()
            list(gen())
            return x + 1

        # A fresh code object: the executor keeps the table with its CFG.
        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__, closure=f.__closure__)
        with self.assertRaisesRegex(ValueError, "hasn't run"):
            torch._rust.cfg_observed(f)
        for _ in range(2):
            self.same(f, torch.ones(2))
        text = re.sub(r"\S+<locals>\.", "", torch._rust.cfg_observed(f, types=False))
        self.assertExpectedInline(
            text,
            """\
bb0:
  CopyFreeVars 3
  Resume
  %0 = LoadDeref py
  %1 = Call 0 null, %0
      callee py
      effects [2 visits]:
        call py
  %2 = LoadDeref via_c
  %3 = Call 0 null, %2
      callee via_c
      effects [2 visits]:
        call via_c
  %4 = LoadGlobal list
  %5 = LoadDeref gen
  %6 = Call 0 null, %5
      callee gen
      effects [2 visits]:
        call gen
  %7 = Call 1 null, %4, %6
      callee builtins.list
      effects [2 visits]:
        resume gen
        resume gen
  %8 = LoadLocal x
  %9 = Const 1
  %10 = BinaryOp + %8, %9
      effects [2 visits]:
        torch aten::add.Tensor(f32[2], i64[], alpha=1)
  return %10
""",
        )
        text = re.sub(
            r"\S+<locals>\.", "", torch._rust.cfg_observed(via_c, types=False)
        )
        lam = "        call <lambda> (indirect)\n"
        self.assertIn(lam + lam, text)

    def test_observed_types(self):
        def f(x, n, opt, k):
            acc = []
            for i in range(n):
                acc += [i]
            y = x if opt is None else opt
            return (acc, "k"), {k: y}

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        p = torch.nn.Parameter(torch.ones(2, 2))
        self.same(f, torch.ones(4), 2, None, 1)
        self.same(f, torch.ones(8), 3, p, 1 << 70)
        self.assertExpectedInline(
            torch._rust.cfg_observed(f, effects=False),
            """\
bb0:
  Resume
  %0: list[0] = BUILD_LIST 0
  StoreLocal acc %0
  %1: type = LoadGlobal range
  %2: int[2..3] = LoadLocal n
  %3: range = Call 1 null, %1, %2
  %4: range_iterator = GetIter %3
  jump bb1(s0=%4)
bb1(%5 s0: range_iterator):
  %6 = IterNext %5
  %7: int[0..2] | null = Project 0 %6
  %8: bool = Project 1 %6
  branch %8, bb2(s0=%5, s1=%7), bb3
bb2(%9 s0: range_iterator, %10 s1: int[0..2]):
  StoreLocal i %10
  %11: list[0..2] = LoadLocal acc
  %12: int[0..2] = LoadLocal i
  %13: list[1] = BUILD_LIST 1 %12
  %14: list[1..3] = BinaryOp += %11, %13
  StoreLocal acc %14
  EvalBreaker
  jump bb1(s0=%9)
bb3:
  %15: None | Parameter(f32 cpu [2, 2] grad) = LoadLocal opt
  %16: bool = IsNone %15
  branch %16, bb4, bb5
bb4:
  %17: Tensor(f32 cpu [4]) = LoadLocal x
  jump bb6(s0=%17)
bb5:
  %18: Parameter(f32 cpu [2, 2] grad) = LoadLocal opt
  jump bb6(s0=%18)
bb6(%19 s0: Tensor(f32 cpu [4]) | Parameter(f32 cpu [2, 2] grad)):
  StoreLocal y %19
  %20: list[2..3] = LoadLocal acc
  %21: str "k" = Const 'k'
  %22: tuple(list[2..3], str "k") = BuildTuple 2 %20, %21
  %23: int[?] = LoadLocal k
  %24: Tensor(f32 cpu [4]) | Parameter(f32 cpu [2, 2] grad) = LoadLocal y
  %25: dict[1] = BUILD_MAP 1 %23, %24
  %26: tuple(tuple[2], dict[1]) = BuildTuple 2 %22, %25
  return %26
""",
        )

    def test_optimize_block(self):
        def f(x, n):
            y = x * 2
            if n:
                y = y + 1
            return y

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        with self.assertRaisesRegex(ValueError, "hasn't run"):
            torch._rust.optimize_block(f, 0)
        self.same(f, torch.ones(2), 1)
        torch._rust.optimize_block(f, 1)
        self.same(f, torch.ones(2), 1)
        self.assertExpectedInline(
            torch._rust.cfg_observed(f),
            """\
bb0:
  Resume
  %0: Tensor(f32 cpu [2]) = LoadLocal x
  %1: int 2 = Const 2
  %2: Tensor(f32 cpu [2]) = BinaryOp * %0, %1
      effects [2 visits]:
        torch aten::mul.Tensor(f32[2], i64[])
  StoreLocal y %2
  %3: int 1 = LoadLocal n
  %4: True = ToBool %3
  branch %4, bb1, bb4
bb1:
  %5: Tensor(f32 cpu [2]) = LoadLocal y
  jump bb2(s0=%5)
bb2(%6 s0: Tensor(f32 cpu [2])):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    %7: int 1 = Const 1
    %8: Tensor(f32 cpu [2]) = Aten aten::add.Tensor %6, %7
        effects [1 visit]:
          torch aten::add.Tensor(f32[2], i64[], alpha=1)
    outputs [%8]
  jump bb3(s0=o0)
  origin:
    bb0(%0 s0):
      %1 = Const 1
      %2 = BinaryOp + %0, %1
      leave(%2)
bb3(%9 s0):
  StoreLocal y %9
  jump bb4
bb4:
  %10: Tensor(f32 cpu [2]) = LoadLocal y
  return %10
""",
        )
        # bb1 is glue now, with nothing left to optimize; bb2 is the pure block.
        torch._rust.optimize_block(f, 1)
        with self.assertRaisesRegex(ValueError, "already has an origin"):
            torch._rust.optimize_block(f, 2)
        with self.assertRaisesRegex(ValueError, "no block bb9"):
            torch._rust.optimize_block(f, 9)

    def test_optimize_block_under_suspended_generator(self):
        def gen(n):
            t = n + 1
            yield t
            yield t * 2

        code = gen.__code__.replace(co_name="gen")
        gen = type(gen)(code, gen.__globals__)
        self.assertEqual(
            torch._rust.call_with_cfg_execution(lambda: list(gen(1))), [2, 4]
        )
        g = gen(3)
        self.assertEqual(torch._rust.call_with_cfg_execution(next, g), 4)
        with self.assertRaisesRegex(ValueError, "generator suspends"):
            torch._rust.optimize_block(gen, 0)
        # The handler block holds no resume point; g resumes under the new CFG.
        torch._rust.optimize_block(gen, 1)
        self.assertEqual(torch._rust.call_with_cfg_execution(list, g), [8])

    # Optimizes every block of fn that can be.
    def test_optimized_stops_recording(self):
        # Once every block was offered to the optimizer, frames record nothing
        # (here the int at bb0's LoadLocal), except in origins run on a miss.
        def f(x):
            y = x * 2
            return y + 1

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        torch._rust._set_cfg_optimize(True)
        try:
            for _ in range(3):
                torch._rust.call_with_cfg_execution(f, torch.ones(2))
            self.assertEqual(torch._rust.call_with_cfg_execution(f, 3), 7)
        finally:
            torch._rust._set_cfg_optimize(False)
        self.assertExpectedInline(
            torch._rust.cfg_observed(f),
            """\
bb0:
  %0: Tensor(f32 cpu [2]) = LoadLocal x
  jump bb1(s0=%0)
bb1(%1 s0):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    Resume
    %2: int 2 = Const 2
    %3: Tensor(f32 cpu [2]) = Aten aten::mul.Tensor %1, %2
        effects [1 visit]:
          torch aten::mul.Tensor(f32[2], i64[])
    %4: int 1 = Const 1
    %5: Tensor(f32 cpu [2]) = Aten aten::add.Tensor %3, %4
        effects [1 visit]:
          torch aten::add.Tensor(f32[2], i64[], alpha=1)
    outputs [%5]
    unwind [4..5) -> bb3(s0=%3)
  jump bb2(s0=o0)
  origin:
    bb0(%0 s0):
      Resume
      %1: int 2 = Const 2
      %2: int 6 = BinaryOp * %0, %1
      %3: int 1 = Const 1
      %4: int 7 = BinaryOp + %2, %3
      leave(%4)
      unwind [4..5) -> outer 0(%2)
bb2(%6 s0):
  return %6
bb3(%7 s0, %8 lasti, %9 exc):
  StoreLocal y %7
  reraise %9, lasti=%8
""",
        )

    def test_effects_vary(self):
        # A site keeps 8 distinct effect sequences.
        def f(x):
            return x + 1

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        for n in range(10):
            torch._rust.call_with_cfg_execution(f, torch.ones(n))
        self.assertExpectedInline(
            torch._rust.cfg_observed(f),
            """\
bb0:
  Resume
  %0: Tensor(f32 cpu [0..9]) = LoadLocal x
  %1: int 1 = Const 1
  %2: Tensor(f32 cpu [0..9]) = BinaryOp + %0, %1
      effects [1 visit]:
        torch aten::add.Tensor(f32[0], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[1], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[2], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[3], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[4], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[5], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[6], i64[], alpha=1)
      effects [1 visit]:
        torch aten::add.Tensor(f32[7], i64[], alpha=1)
      effects vary (more sequences than kept)
  return %2
""",
        )

    def optimize_all(self, fn):
        b = 0
        while True:
            try:
                torch._rust.optimize_block(fn, b)
            except ValueError as e:
                if "no block" in str(e):
                    return
            b += 1

    def test_callees_dont_keep_objects_alive(self):
        # Through a proxy, p.copy is a builtin bound to the set; a class is
        # recorded through a weakref.
        def f(p):
            return p.copy()

        def g(cls):
            return cls()

        class A:
            pass

        s = {1}
        p = weakref.proxy(s)
        self.assertEqual(torch._rust.call_with_cfg_execution(f, p), {1})
        r = weakref.ref(A)
        torch._rust.call_with_cfg_execution(g, A)
        self.assertIn("callee", torch._rust.cfg_observed(g))
        del s, A
        gc.collect()
        with self.assertRaises(ReferenceError):
            len(p)
        self.assertIsNone(r())

    def test_promote_locals(self):
        def f(x):
            y = x * 2
            y = y + x
            z = y * y
            return z + y

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        self.same(f, 3)
        self.optimize_all(f)
        self.same(f, 3)
        self.assertExpectedInline(
            torch._rust.cfg_observed(f),
            """\
bb0:
  %0: int 3 = LoadLocal x
  jump bb1(s0=%0)
bb1(%1 s0: int 3):
  Resume
  %2: int 2 = Const 2
  %3: int 6 = BinaryOp * %1, %2
  %4: int 9 = BinaryOp + %3, %1
  %5: int 81 = BinaryOp * %4, %4
  %6: int 90 = BinaryOp + %5, %4
  jump bb2(s0=%6)
  unwind [3..4) -> bb3(s0=%3)
  unwind [4..5) -> bb4(s0=%4)
  unwind [5..6) -> bb5(s0=%4, s1=%5)
  origin:
    bb0(%0 s0):
      Resume
      %1 = Const 2
      %2 = BinaryOp * %0, %1
      %3 = BinaryOp + %2, %0
      %4 = BinaryOp * %3, %3
      %5 = BinaryOp + %4, %3
      leave(%5)
      unwind [3..4) -> outer 0(%2)
      unwind [4..5) -> outer 1(%3)
      unwind [5..6) -> outer 2(%3, %4)
bb2(%7 s0):
  return %7
bb3(%8 s0, %9 lasti, %10 exc):
  StoreLocal y %8
  reraise %10, lasti=%9
bb4(%11 s0, %12 lasti, %13 exc):
  StoreLocal y %11
  reraise %13, lasti=%12
bb5(%14 s0, %15 s1, %16 lasti, %17 exc):
  StoreLocal y %14
  StoreLocal z %15
  reraise %17, lasti=%16
""",
        )

    def test_hoist_loads(self):
        # A block's first load of each local moves to its entry.
        def f(a, b, c):
            return a * b + c

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        self.same(f, 1, 2, 3)
        self.optimize_all(f)
        self.same(f, 1, 2, 3)
        version = torch._rust.cfg_observed(f).split("  origin:")[0]
        self.assertExpectedInline(
            version,
            """\
bb0:
  %0: int 1 = LoadLocal a
  %1: int 2 = LoadLocal b
  %2: int 3 = LoadLocal c
  jump bb1(s0=%0, s1=%1, s2=%2)
bb1(%3 s0: int 1, %4 s1: int 2, %5 s2: int 3):
  Resume
  %6: int 2 = BinaryOp * %3, %4
  %7: int 5 = BinaryOp + %6, %5
  jump bb2(s0=%7)
""",
        )

    def test_aten_replacement(self):
        # Operators become the aten ops they ran, guarded on the params' types
        # and the torch-function state; when the guard fails the origin runs
        # the Python-level operators.
        def f(a, b, n):
            return -(a * b + n) > 2

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        a, b = torch.tensor([1.0, 2.0]), torch.tensor([3.0, -4.0])
        self.same(f, a, b, 1)
        self.optimize_all(f)
        self.same(f, a, b, 1)
        version = torch._rust.cfg_observed(f, effects=False).split("  origin:")[0]
        self.assertExpectedInline(
            version,
            """\
bb0:
  %0: Tensor(f32 cpu [2]) = LoadLocal a
  %1: Tensor(f32 cpu [2]) = LoadLocal b
  %2: int 1 = LoadLocal n
  jump bb1(s0=%0, s1=%1, s2=%2)
bb1(%3 s0: Tensor(f32 cpu [2]), %4 s1: Tensor(f32 cpu [2]), %5 s2: int 1):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    guard s1 is a plain tensor
    guard type(s2) is int
    Resume
    %6: Tensor(f32 cpu [2]) = Aten aten::mul.Tensor %3, %4
    %7: Tensor(f32 cpu [2]) = Aten aten::add.Tensor %6, %5
    %8: Tensor(f32 cpu [2]) = Aten aten::neg %7
    %9: int 2 = Const 2
    %10: Tensor(b8 cpu [2]) = Aten aten::gt.Scalar %8, %9
    outputs [%10]
  jump bb2(s0=o0)
""",
        )

        class Sub(torch.Tensor):
            def __mul__(self, other):
                return torch.zeros(2)

        class Mode(torch.overrides.TorchFunctionMode):
            calls = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                Mode.calls += 1
                return func(*args, **(kwargs or {}))

        # A subclass, a float where an int was guarded, and an active mode each
        # miss the guard, so the operators run as Python would run them.
        self.same(f, a.as_subclass(Sub), b, 1)
        self.same(f, a, b, 1.5)
        with Mode():
            torch._rust.call_with_cfg_execution(f, a, b, 1)
        self.assertEqual(Mode.calls, 4)
        self.same(f, a, b, 1)

    def test_guard_checks(self):
        # Each check misses (running the origin) exactly when its condition
        # fails.
        def f(a, b, n):
            return a * b + n

        a, b = torch.randn(2, 3), torch.randn(2, 3)
        self.same(f, a, b, 1)
        self.optimize_all(f)
        checks = [
            ("grad_enabled", True),
            ("requires_grad", 0, False),
            ("tensor", 0, a),
            ("value", 2, 1),
            ("no_forward_ad",),
            ("no_saved_tensor_hooks",),
        ]
        torch._rust._add_block_checks(f, 1, checks)
        version = torch._rust.cfg_observed(f, effects=False, types=False).split(
            "  origin:"
        )[0]
        self.assertExpectedInline(
            version,
            """\
bb0:
  %0 = LoadLocal a
  %1 = LoadLocal b
  %2 = LoadLocal n
  jump bb1(s0=%0, s1=%1, s2=%2)
bb1(%3 s0, %4 s1, %5 s2):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    guard s1 is a plain tensor
    guard type(s2) is int
    guard grad enabled
    guard not s0.requires_grad
    guard s0 is tensor(f32 cpu [2, 3] stride [3, 1])
    guard s2 == 1
    guard no forward AD
    guard no saved tensor hooks
    Resume
    %6 = Aten aten::mul.Tensor %3, %4
    %7 = Aten aten::add.Tensor %6, %5
    outputs [%7]
  jump bb2(s0=o0)
""",
        )

        def misses(*args):
            before = torch._rust._cfg_misses(f)[1]
            self.assertEqual(torch._rust.call_with_cfg_execution(f, *args), f(*args))
            return torch._rust._cfg_misses(f)[1] - before

        self.assertEqual(misses(a, b, 1), 0)
        with torch.no_grad():
            self.assertEqual(misses(a, b, 1), 1)
        self.assertEqual(misses(a.clone().requires_grad_(), b, 1), 1)
        self.assertEqual(misses(torch.randn(3, 2), torch.randn(3, 2), 1), 1)
        self.assertEqual(misses(torch.randn(3, 2).t(), b, 1), 1)
        self.assertEqual(misses(a.double(), b.double(), 1), 1)
        self.assertEqual(misses(a, b, 2), 1)
        self.assertEqual(misses(a, b, True), 1)
        with torch.autograd.forward_ad.dual_level():
            self.assertEqual(misses(a, b, 1), 1)
        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            self.assertEqual(misses(a, b, 1), 1)
        self.assertEqual(misses(a, b, 1), 0)

    def test_optimized_grads_match_eager(self):
        # Replaced operators still run through autograd, with grad on or off
        # and whichever inputs require grad.
        def f(a, b, c):
            return ((a * b + c) @ b.t() - a).sum()

        def grads(fn, run, *args):
            args = [x.clone().requires_grad_(rg) for x, rg in args]
            out = run(fn, *args)
            if not out.requires_grad:
                return out, []
            out.backward()
            return out, [x.grad for x in args]

        x, y, z = (torch.randn(3, 3) for _ in range(3))
        self.same(f, x, y, z)
        self.optimize_all(f)
        self.assertIn(
            "Aten aten::mul.Tensor",
            torch._rust.cfg_observed(f, effects=False, types=False),
        )
        eager = lambda fn, *args: fn(*args)  # noqa: E731
        for rgs in [
            (True, True, True),
            (True, False, False),
            (False, False, True),
            (False, False, False),
        ]:
            args = list(zip((x, y, z), rgs))
            self.assertEqual(
                grads(f, torch._rust.call_with_cfg_execution, *args),
                grads(f, eager, *args),
            )
            with torch.no_grad():
                self.assertEqual(
                    grads(f, torch._rust.call_with_cfg_execution, *args),
                    grads(f, eager, *args),
                )

    def test_aten_graph(self):
        # An op list becomes a graph of aten ops: kwargs by name, literals
        # (deduplicated), lists of values built at run time, and multi-output
        # ops (tuples and Tensor[]) through Projects. It runs without a frame.
        from torch._rust import Ref

        x, w = torch.randn(4, 3), torch.randn(3)
        ops = [
            ("aten::mul.Tensor", (Ref(0), Ref(1)), {}, None),
            ("aten::add.Tensor", (Ref(2), 1.5), {"alpha": 2}, None),
            ("aten::var_mean.correction", (Ref(3), [1]), {"keepdim": True}, 2),
            ("aten::split.Tensor", (Ref(3), 2), {}, 2),
            ("aten::cat", ([Ref(6), Ref(7)], 0), {}, None),
            (
                "aten::sum.dim_IntList",
                (Ref(8), [0], False),
                {"dtype": torch.float64},
                None,
            ),
            (
                "aten::zeros",
                ([2, 2],),
                {
                    "dtype": torch.float32,
                    "layout": torch.strided,
                    "device": torch.device("cpu"),
                },
                None,
            ),
        ]
        text, outs = torch._rust._run_aten_graph(
            2, ops, [Ref(4), Ref(5), Ref(8), Ref(9), Ref(10)], [x, w]
        )
        self.assertExpectedInline(
            text,
            """\
bb0(%0 s0, %1 s1):
  %2 = Aten aten::mul.Tensor %0, %1
  %3 = Lit 1.5
  %4 = Lit 2
  %5 = Aten aten::add.Tensor kw[alpha] %2, %3, %4
  %6 = Lit [1]
  %7 = Lit True
  %8 = Aten aten::var_mean.correction kw[keepdim] %5, %6, %7
  %9 = Project 0 %8
  %10 = Project 1 %8
  %11 = Aten aten::split.Tensor %5, %4
  %12 = Project 0 %11
  %13 = Project 1 %11
  %14 = BuildTuple 2 %12, %13
  %15 = Lit 0
  %16 = Aten aten::cat %14, %15
  %17 = Lit [0]
  %18 = Lit False
  %19 = Lit torch.float64
  %20 = Aten aten::sum.dim_IntList kw[dtype] %16, %17, %18, %19
  %21 = Lit [2, 2]
  %22 = Lit torch.float32
  %23 = Lit torch.strided
  %24 = Lit device("cpu")
  %25 = Aten aten::zeros kw[dtype, layout, device] %21, %22, %23, %24
  leave(%9, %10, %16, %20, %25)
""",
        )
        t = torch.add(x * w, 1.5, alpha=2)
        cat = torch.cat(t.split(2))
        self.assertEqual(
            outs,
            (
                *torch.var_mean(t, [1], keepdim=True),
                cat,
                cat.sum([0], dtype=torch.float64),
                torch.zeros(2, 2),
            ),
        )

        mm = [("aten::mm", (Ref(0), Ref(0)), {}, None)]
        with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
            torch._rust._run_aten_graph(1, mm, [Ref(1)], [torch.randn(2, 3)])
        with self.assertRaisesRegex(ValueError, "not defined yet"):
            torch._rust._run_aten_graph(1, mm, [Ref(2)], [torch.randn(2, 3)])
        with self.assertRaisesRegex(ValueError, "as a literal"):
            torch._rust._run_aten_graph(
                1, [("aten::mul.Tensor", (Ref(0), object()), {}, None)], [Ref(1)], [x]
            )

    def test_joint_graph(self):
        # The forward and backward ops below Autograd as one graph: tensors
        # autograd saved (sigmoid's and exp's outputs) map back to the values
        # that made them, and an in-place op on an intermediate becomes its
        # out-of-place overload.
        def f(x, w, c):
            y = (x @ w).sigmoid()
            z = y * c
            z.add_(1)
            return z.exp().sum(), y

        x, w, c = (
            torch.randn(4, 3, requires_grad=True),
            torch.randn(3, 5, requires_grad=True),
            torch.randn(4, 5),
        )
        self.assertExpectedInline(
            torch._rust._trace_joint(f, x, w, c),
            """\
# 3 inputs, 2 tangents, 2 outputs; the first 9 instrs are the forward
bb0(%0 s0, %1 s1, %2 s2, %3 s3, %4 s4):
  %5 = Aten aten::mm %0, %1
  %6 = Aten aten::sigmoid %5
  %7 = Aten aten::detach %6
  %8 = Aten aten::mul.Tensor %6, %2
  %9 = Lit 1
  %10 = Aten aten::add.Tensor %8, %9
  %11 = Aten aten::exp %10
  %12 = Aten aten::detach %11
  %13 = Aten aten::sum %11
  %14 = Lit [4, 5]
  %15 = Aten aten::expand %3, %14
  %16 = Aten aten::detach %12
  %17 = Aten aten::mul.Tensor %15, %16
  %18 = Aten aten::mul.Tensor %17, %2
  %19 = Aten aten::add.Tensor %4, %18
  %20 = Aten aten::detach %7
  %21 = Aten aten::sigmoid_backward %19, %20
  %22 = Aten aten::t %0
  %23 = Aten aten::mm %22, %21
  %24 = Aten aten::t %1
  %25 = Aten aten::mm %21, %24
  leave(%13, %6, %25, %23)
""",
        )
        self.assertExpectedInline(
            torch._rust._trace_joint(lambda x: torch.softmax(x, -1).relu(), x),
            """\
# 1 inputs, 1 tangents, 1 outputs; the first 6 instrs are the forward
bb0(%0 s0, %1 s1):
  %2 = Lit -1
  %3 = Lit False
  %4 = Aten aten::_softmax %0, %2, %3
  %5 = Aten aten::detach %4
  %6 = Aten aten::relu %4
  %7 = Aten aten::detach %6
  %8 = Aten aten::detach %7
  %9 = Lit 0
  %10 = Aten aten::threshold_backward %1, %8, %9
  %11 = Aten aten::detach %5
  %12 = Lit torch.float32
  %13 = Aten aten::_softmax_backward_data %10, %11, %2, %12
  leave(%6, %13)
""",
        )

        def mutate_view(x):
            y = x * 2
            y.view(-1).add_(1)
            return y

        def mutate_input(x, c):
            c.add_(1)
            return x * c

        refused = [
            (lambda x: torch.dropout(x, 0.5, True), (x,), "bernoulli_.float is random"),
            (
                lambda x: x[x > 0],
                (x,),
                "index.Tensor has a data-dependent output shape",
            ),
            (mutate_view, (x,), "add_.Tensor mutates a view or a viewed tensor"),
            (mutate_input, (x, torch.randn(4, 3)), "add_.Tensor mutates an input"),
            (
                lambda x: x * c[:, :3],
                (x,),
                "slice.Tensor uses a tensor from outside the graph",
            ),
        ]
        for fn, args, msg in refused:
            with self.assertRaisesRegex(NotImplementedError, msg):
                torch._rust._trace_joint(fn, *args)

    def test_joint_partition(self):
        # The min-cut partition saves what can't be recomputed (mm's output)
        # and recomputes pointwise ops and views in the backward, from the
        # inputs a node keeps anyway.
        def mlp(x, w):
            h = x @ w
            return (h.sigmoid() * h).exp().sum()

        x, w = (
            torch.randn(8, 4, requires_grad=True),
            torch.randn(4, 2, requires_grad=True),
        )
        self.assertExpectedInline(
            torch._rust._trace_joint(mlp, x, w, partition=True),
            """\
forward:
bb0(%0 s0, %1 s1):
  %2 = Aten aten::mm %0, %1
  %3 = Aten aten::sigmoid %2
  %4 = Aten aten::mul.Tensor %3, %2
  %5 = Aten aten::exp %4
  %6 = Aten aten::sum %5
  leave(%6, %0, %1, %2)
backward:
bb0(%0 s0, %1 s1, %2 s2, %3 s3):
  %4 = Aten aten::sigmoid %2
  %5 = Aten aten::detach %4
  %6 = Aten aten::mul.Tensor %4, %2
  %7 = Aten aten::exp %6
  %8 = Aten aten::detach %7
  %9 = Lit [8, 2]
  %10 = Aten aten::expand %3, %9
  %11 = Aten aten::detach %8
  %12 = Aten aten::mul.Tensor %10, %11
  %13 = Aten aten::mul.Tensor %12, %4
  %14 = Aten aten::mul.Tensor %12, %2
  %15 = Aten aten::detach %5
  %16 = Aten aten::sigmoid_backward %14, %15
  %17 = Aten aten::add.Tensor %13, %16
  %18 = Aten aten::t %0
  %19 = Aten aten::mm %18, %17
  %20 = Aten aten::t %1
  %21 = Aten aten::mm %17, %20
  leave(%21, %19)
""",
        )

    def test_autograd_bootstrap(self):
        # A pure block of aten ops gets its own autograd node; its backward
        # runs autograd on a traced copy of the block's run, recording the
        # joint graph, and matches eager.
        def f(a, b, c):
            return (a * b + c) * a - b, a, a + c

        x, y, z = torch.randn(3), torch.randn(3), torch.randn(3)
        self.same(f, x, y, z)
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:

            def run(fn, rgs, use=(0, 2)):
                args = [t.clone().requires_grad_(rg) for t, rg in zip((x, y, z), rgs)]
                out = fn(f, *args)
                self.assertIs(out[1], args[0])
                sum(out[i].sum() for i in use).backward()
                return out, [a.grad for a in args]

            cfg = lambda fn, *a: torch._rust.call_with_cfg_execution(fn, *a)  # noqa: E731
            eager = lambda fn, *a: fn(*a)  # noqa: E731
            for rgs in [(True, True, True), (True, False, False), (False, True, True)]:
                self.assertEqual(run(cfg, rgs), run(eager, rgs))
            # An output that gets no gradient.
            self.assertEqual(
                run(cfg, (True, True, True), use=(0,)),
                run(eager, (True, True, True), use=(0,)),
            )
            self.assertExpectedInline(
                torch._rust._cfg_last_joint(),
                """\
bb0(%0 s0, %1 s1, %2 s2, %3 s3, %4 s4):
  %5 = Aten aten::mul.Tensor %0, %1
  %6 = Aten aten::add.Tensor %5, %2
  %7 = Aten aten::mul.Tensor %6, %0
  %8 = Aten aten::sub.Tensor %7, %1
  %9 = Aten aten::add.Tensor %0, %2
  %10 = Aten aten::neg %3
  %11 = Aten aten::mul.Tensor %3, %6
  %12 = Aten aten::mul.Tensor %3, %0
  %13 = Aten aten::add.Tensor %4, %11
  %14 = Aten aten::add.Tensor %4, %12
  %15 = Aten aten::mul.Tensor %12, %0
  %16 = Aten aten::mul.Tensor %12, %1
  %17 = Aten aten::add.Tensor %13, %16
  %18 = Aten aten::add.Tensor %10, %15
  leave(%8, %9, %17, %18, %14)
""",
            )

            def outputs():
                a, b, c = (t.clone().requires_grad_() for t in (x, y, z))
                return (a, b, c), torch._rust.call_with_cfg_execution(f, a, b, c)

            (a, b, c), out = outputs()
            self.assertEqual(out[0].grad_fn.name(), "CfgBackward")
            with torch.no_grad():
                b.add_(1)
            with self.assertRaisesRegex(
                RuntimeError, "modified by an inplace operation"
            ):
                out[0].sum().backward()

            _, out = outputs()
            out[0].sum().backward(retain_graph=True)
            out[0].sum().backward()
            with self.assertRaisesRegex(
                RuntimeError, "backward through the graph a second time"
            ):
                out[0].sum().backward()

            # Double backward differentiates through the real inputs.
            def second(fn):
                args = [t.clone().requires_grad_() for t in (x, y, z)]
                out = fn(f, *args)
                (g,) = torch.autograd.grad(
                    (out[0] * out[0]).sum(), args[0], create_graph=True
                )
                return torch.autograd.grad(g.sum(), args[1])

            self.assertEqual(second(cfg), second(eager))

            with torch.no_grad():
                self.assertFalse(outputs()[1][0].requires_grad)
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_autograd_training_version(self):
        # The first backward splits the joint graph into a forward that saves
        # what the backward needs and a backward, run by a version guarded on
        # what the trace depended on; other runs take the eager-autograd path.
        def f(a, b, n):
            return (a * b + n) * a

        x, y = torch.randn(3), torch.randn(3)
        self.same(f, x, y, 2)
        self.optimize_all(f)
        cfg = lambda *args: torch._rust.call_with_cfg_execution(f, *args)  # noqa: E731

        def grads(fn, a, b, n, create_graph=False):
            a, b = a.clone().requires_grad_(), b.clone().requires_grad_()
            out = fn(a, b, n)
            if create_graph:
                (g,) = torch.autograd.grad(out.sum(), a, create_graph=True)
                return torch.autograd.grad(g.sum(), b)
            out.sum().backward()
            return out, a.grad, b.grad, out.grad_fn.name()

        torch._rust._set_cfg_autograd(True)
        try:
            # A forward still pending when the first backward swaps in the
            # training version keeps its own backward.
            a1, b1, a2, b2 = (t.clone().requires_grad_() for t in (x, y, x, y))
            o1, o2 = cfg(a1, b1, 2), cfg(a2, b2, 2)
            o1.sum().backward()
            o2.sum().backward()
            self.assertEqual((a1.grad, b1.grad), (a2.grad, b2.grad))
            self.assertExpectedInline(
                torch._rust.cfg_observed(f, effects=False, types=False).split(
                    "  origin:"
                )[0],
                """\
bb0:
  %0 = LoadLocal a
  %1 = LoadLocal b
  %2 = LoadLocal n
  jump bb1(s0=%0, s1=%1, s2=%2)
bb1(%3 s0, %4 s1, %5 s2):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    guard s1 is a plain tensor
    guard type(s2) is int
    guard grad enabled
    guard no forward AD
    guard no saved tensor hooks
    guard s0.requires_grad
    guard s0 is tensor(f32 cpu [3] stride [1])
    guard s1.requires_grad
    guard s1 is tensor(f32 cpu [3] stride [1])
    guard s2 == 2
    %6 = Autograd t0 %3, %4, %5
    %7 = Project 0 %6
    outputs [%7]
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    guard s1 is a plain tensor
    guard type(s2) is int
    Resume
    %8 = Aten aten::mul.Tensor %3, %4
    %9 = Aten aten::add.Tensor %8, %5
    %10 = Aten aten::mul.Tensor %9, %3
    outputs [%10]
  jump bb2(s0=o0)
""",
            )
            self.assertExpectedInline(
                torch._rust.cfg_observed(f, effects=False, types=False).split(
                    "t0 forward:"
                )[1],
                """\

    bb0(%0 s0, %1 s1):
      %2 = Aten aten::mul.Tensor %0, %1
      %3 = Lit 2
      %4 = Aten aten::add.Tensor %2, %3
      %5 = Aten aten::mul.Tensor %4, %0
      leave(%5, %0, %1)
t0 backward:
    bb0(%0 s0, %1 s1, %2 s2):
      %3 = Aten aten::mul.Tensor %0, %1
      %4 = Lit 2
      %5 = Aten aten::add.Tensor %3, %4
      %6 = Aten aten::mul.Tensor %2, %5
      %7 = Aten aten::mul.Tensor %2, %0
      %8 = Aten aten::mul.Tensor %7, %0
      %9 = Aten aten::mul.Tensor %7, %1
      %10 = Aten aten::add.Tensor %6, %9
      leave(%10, %8)
""",
            )
            out, *rest = grads(cfg, x, y, 2)
            self.assertEqual(rest[-1], "CfgBackward")
            self.assertEqual((out, *rest[:-1]), grads(f, x, y, 2)[:-1])
            self.assertEqual(
                grads(cfg, x, y, 2, create_graph=True),
                grads(f, x, y, 2, create_graph=True),
            )
            # Guard misses: another shape, dtype or scalar value.
            for args in [
                (torch.randn(4), torch.randn(4), 2),
                (x.double(), y.double(), 2),
                (x, y, 3),
                (x, y, 2.0),
            ]:
                got, want = grads(cfg, *args), grads(f, *args)
                self.assertEqual(got, want)

            a, b = x.clone().requires_grad_(), y.clone().requires_grad_()
            out = cfg(a, b, 2)
            with torch.no_grad():
                a.add_(1)
            with self.assertRaisesRegex(
                RuntimeError, "modified by an inplace operation"
            ):
                out.sum().backward()
            out = cfg(x.clone().requires_grad_(), y.clone().requires_grad_(), 2)
            out.sum().backward()
            with self.assertRaisesRegex(
                RuntimeError, "backward through the graph a second time"
            ):
                out.sum().backward()
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_autograd_untrainable(self):
        # A block whose params a guard can't pin down (here a dict) gets no
        # training version, so after its first backward it stops bootstrapping
        # and takes the eager-autograd path.
        def f(a, b, d):
            return a * b, d

        x, y, d = torch.randn(3), torch.randn(3), {}
        self.same(f, x, y, d)
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:
            names = []
            for _ in range(3):
                a, b = x.clone().requires_grad_(), y.clone().requires_grad_()
                out, dd = torch._rust.call_with_cfg_execution(f, a, b, d)
                self.assertIs(dd, d)
                names.append(out.grad_fn.name())
                out.sum().backward()
                self.assertEqual((a.grad, b.grad), (y, x))
            self.assertEqual(names, ["CfgBackward", "MulBackward0", "MulBackward0"])
            self.assertEqual(
                torch._rust._cfg_autograd_blocks(f),
                [(1, 0, "gave up: param 2 is a dict, which a guard can't pin down")],
            )
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_autograd_null_param(self):
        # A pure block can carry CPython's NULL through (here the self slot
        # of calling an attribute that isn't a method, pushed before the
        # args are computed), along with the callable: its guard checks the
        # NULL, the callable is only passed on, and both are outputs of its
        # training version.
        class M:
            pass

        m = M()
        m.o = lambda t: t.sum()

        def f(m, a, b):
            return m.o(a @ b * 2)

        x, y = torch.randn(3, 3), torch.randn(3, 3)
        self.same(f, m, x, y)
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(3):
                a, b = x.clone().requires_grad_(), y.clone().requires_grad_()
                torch._rust.call_with_cfg_execution(f, m, a, b).backward()
                ones = torch.ones(3, 3)
                self.assertEqual((a.grad, b.grad), (2 * ones @ y.T, 2 * x.T @ ones))
            self.assertEqual(
                torch._rust._cfg_autograd_blocks(f),
                [(1, 0, "trained")],
            )
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_pinned_callee(self):
        # A torch function read from a global through module attributes is
        # pinned; rebinding the attribute misses the guard.
        m = types.ModuleType("m")
        m.act = torch.relu
        ns = {"m": m}
        exec("def f(x):\n    return m.act(x * 2)\n", ns)
        f = ns["f"]
        x = torch.tensor([-1.0, 2.0])
        self.same(f, x)
        self.optimize_all(f)
        out = torch._rust.cfg_observed(f)
        self.assertIn("guard m.act is torch._VariableFunctionsClass.relu", out)
        self.assertIn("Pinned torch._VariableFunctionsClass.relu", out)
        self.assertEqual(
            torch._rust.call_with_cfg_execution(f, x), torch.tensor([0.0, 4.0])
        )
        self.assertEqual(torch._rust._cfg_misses(f), [0, 0, 0])
        m.act = torch.neg
        self.assertEqual(
            torch._rust.call_with_cfg_execution(f, x), torch.tensor([2.0, -4.0])
        )
        self.assertEqual(torch._rust._cfg_misses(f), [0, 1, 0])

    def test_autograd_pinned_call(self):
        # A pinned call that ran one torch op trains like an aten op, and its
        # result counts as a plain tensor for the operators after it.
        def f(x, w):
            return torch.relu(x @ w) * 2

        x, w = torch.randn(3, 3), torch.randn(3, 3)
        self.same(f, x, w)
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(3):
                a, b = x.clone().requires_grad_(), w.clone().requires_grad_()
                torch._rust.call_with_cfg_execution(f, a, b).sum().backward()
                a2, b2 = x.clone().requires_grad_(), w.clone().requires_grad_()
                f(a2, b2).sum().backward()
                self.assertEqual((a.grad, b.grad), (a2.grad, b2.grad))
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
            training = torch._rust.cfg_observed(f).split("Autograd")[0]
            relu = "torch._VariableFunctionsClass.relu"
            self.assertIn(f"guard torch.relu is {relu}", training)
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_autograd_pinned_python_leaf(self):
        # A Python function Dynamo puts in its graphs (F.relu) whose own frame
        # ran only torch ops is pinned like a builtin, so the operators around
        # its call land in the same trained block.
        def f(x, w):
            return F.relu(x @ w) * 2

        x, w = torch.randn(3, 3), torch.randn(3, 3)
        self.same(f, x, w)
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(3):
                a, b = x.clone().requires_grad_(), w.clone().requires_grad_()
                torch._rust.call_with_cfg_execution(f, a, b).sum().backward()
                a2, b2 = x.clone().requires_grad_(), w.clone().requires_grad_()
                f(a2, b2).sum().backward()
                self.assertEqual((a.grad, b.grad), (a2.grad, b2.grad))
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
            out = torch._rust.cfg_observed(f)
            self.assertIn("guard F.relu is torch.nn.functional.relu", out)
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_reoptimize(self):
        # Optimizing again starts from the unoptimized CFG: the result is what
        # the first optimization made (nothing new was observed), pinned
        # callees and all, and it still trains.
        def f(x, w):
            return F.relu(x @ w) * 2

        x, w = torch.randn(3, 3), torch.randn(3, 3)
        self.same(f, x, w)
        self.optimize_all(f)
        first = torch._rust.cfg_observed(f, False, False)
        torch._rust._reoptimize(f)
        self.assertEqual(torch._rust.cfg_observed(f, False, False), first)
        self.assertEqual(torch._rust.call_with_cfg_execution(f, x, w), f(x, w))
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(3):
                a = w.clone().requires_grad_()
                torch._rust.call_with_cfg_execution(f, x, a).sum().backward()
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_clones(self):
        # After optimizing, each call site of a Python function gets its own
        # clone of the callee's CFG, which records only that site's visits.
        def g(a):
            return a + 1

        def f(x, y):
            return g(x), g(y)

        self.same(f, 1, 2.0)
        self.optimize_all(f)
        clones = torch._rust._cfg_clones(f)
        self.assertEqual([q for _, q in clones], [g.__qualname__] * 2)
        self.assertEqual(torch._rust.call_with_cfg_execution(f, 3, 4.5), (4, 5.5))
        (v1, _), (v2, _) = clones
        self.assertIn("int 4 = BinaryOp +", torch._rust.cfg_observed(f, clone=[v1]))
        self.assertIn("float = BinaryOp +", torch._rust.cfg_observed(f, clone=[v2]))
        self.assertEqual(torch._rust.call_with_cfg_execution(f, 1.5, 2), (2.5, 3))

        # A cloned site that calls another function runs that function.
        def h1(a):
            return a + 1

        def h2(a):
            return a * 10

        def k(h, a):
            return h(a)

        self.same(k, h1, 1)
        self.optimize_all(k)
        self.assertEqual([q for _, q in torch._rust._cfg_clones(k)], [h1.__qualname__])
        self.assertEqual(torch._rust.call_with_cfg_execution(k, h2, 2), 20)
        self.assertEqual(torch._rust.call_with_cfg_execution(k, h1, 2), 3)

    def test_guard_miss_and_raise(self):
        # A check in a version's guard that fails is a miss: the block's
        # origin runs. A raise in the body unwinds from the version, with
        # CPython's exception and traceback line.
        def f(x, w):
            return x @ w * 2

        x, w = torch.randn(3, 3), torch.randn(3, 3)
        self.same(f, x, w)
        self.optimize_all(f)
        torch._rust._add_block_checks(f, 1, [("expect_is", 1, w)])
        self.assertIn("guard s1 is tensor([[", torch._rust.cfg_observed(f))
        self.assertEqual(torch._rust.call_with_cfg_execution(f, x, w), f(x, w))
        self.assertEqual(torch._rust._cfg_misses(f)[1], 0)
        w2 = torch.randn(3, 3)
        self.assertEqual(torch._rust.call_with_cfg_execution(f, x, w2), f(x, w2))
        self.assertEqual(torch._rust._cfg_misses(f)[1], 1)
        try:
            torch._rust.call_with_cfg_execution(f, torch.randn(2, 2), w)
            self.fail("no exception")
        except RuntimeError as e:
            self.assertIn("cannot be multiplied", str(e))
            tb = traceback.extract_tb(e.__traceback__)
        self.assertEqual(tb[-1].name, "f")
        self.assertEqual(tb[-1].line, "return x @ w * 2")
        self.assertEqual(torch._rust._cfg_misses(f)[1], 1)

    def test_elide(self):
        # A call of a Python function whose clone's path can be copied in is
        # elided: its body joins the caller's pure block, behind checks in its
        # guard that the callee and its defaults are the ones observed;
        # a check failing runs the block's origin (the real call).
        ns = {"torch": torch}
        exec(
            "def helper(t, s=2):\n    return torch.relu(t) * s\n\n"
            "def f(x, w):\n    return helper(x @ w) + 1\n",
            ns,
        )
        f = ns["f"]
        x, w = torch.randn(3, 3), torch.randn(3, 3)
        expected = f(x, w)
        torch._rust.call_with_cfg_execution(f, x, w)
        self.optimize_all(f)
        torch._rust.call_with_cfg_execution(f, x, w)
        torch._rust._set_cfg_elide(True)
        try:
            torch._rust._reoptimize(f)
        finally:
            torch._rust._set_cfg_elide(False)
        self.assertExpectedInline(
            torch._rust.cfg_observed(f, False, False).split("origin:")[0],
            """\
bb0:
  %0 = LoadLocal x
  %1 = LoadLocal w
  jump bb1(s0=%0, s1=%1)
bb1(%2 s0, %3 s1):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    guard s1 is a plain tensor
    guard helper is helper
    guard helper.__defaults__ is __defaults__
    guard torch.relu is torch._VariableFunctionsClass.relu
    %4 = Pinned torch._VariableFunctionsClass.relu
    Resume
    %5 = Aten aten::matmul %2, %3
    %6 = Pinned 2
    %7 = Call 1 null, %4, %5
    %8 = Aten aten::mul.Tensor %7, %6
    %9 = Const 1
    %10 = Aten aten::add.Tensor %8, %9
    outputs [%10]
  jump bb2(s0=o0)
  """,
        )
        run = lambda *a: torch._rust.call_with_cfg_execution(f, *a)  # noqa: E731
        self.assertEqual(run(x, w), expected)
        self.assertEqual(torch._rust._cfg_misses(f)[1], 0)
        # A different default: the defaults check fails.
        helper, defaults = ns["helper"], ns["helper"].__defaults__
        helper.__defaults__ = (3,)
        self.assertEqual(run(x, w), torch.relu(x @ w) * 3 + 1)
        self.assertEqual(torch._rust._cfg_misses(f)[1], 1)
        # A different callee.
        ns["helper"] = lambda t: t
        self.assertEqual(run(x, w), x @ w + 1)
        self.assertEqual(torch._rust._cfg_misses(f)[1], 2)
        # A raise in the elided code raises from the caller (the callee's
        # frame isn't in the traceback).
        ns["helper"], helper.__defaults__ = helper, defaults
        self.assertEqual(run(x, w), expected)
        c = torch.randn(3, 3, dtype=torch.complex64)
        try:
            run(c, c)
            self.fail("no exception")
        except RuntimeError as e:
            tb = traceback.extract_tb(e.__traceback__)
            self.assertEqual(tb[-1].name, "f")
        self.assertEqual(torch._rust._cfg_misses(f)[1], 2)

    def elide(self, f, *args):
        # Runs f, optimizes it (giving its call sites clones), runs it again
        # (so the clones observe), and optimizes it again with elision on.
        torch._rust.call_with_cfg_execution(f, *args)
        self.optimize_all(f)
        torch._rust.call_with_cfg_execution(f, *args)
        torch._rust._set_cfg_elide(True)
        try:
            torch._rust._reoptimize(f)
        finally:
            torch._rust._set_cfg_elide(False)

    def test_elide_kwargs_dicts(self):
        # Each elided call's **kwargs is its own dict: guards share reads,
        # but an empty dict is a new object, not a read.
        ns = {"torch": torch}
        exec(
            "def helper(t, **kw):\n    return t * 2, kw\n\n"
            "def f(x):\n    return helper(x), helper(x)\n",
            ns,
        )
        f = ns["f"]
        x = torch.randn(3)
        self.elide(f, x)
        self.assertIn("BUILD_MAP", torch._rust.cfg_observed(f, False, False))
        (_, a), (_, b) = torch._rust.call_with_cfg_execution(f, x)
        self.assertIsNot(a, b)

    def test_elide_autograd(self):
        # An elided block trains: its training version checks its guard,
        # then runs its node's forward on the params and its inputs; a tensor
        # input (W) gets checks in the guard and its gradient.
        ns = {"torch": torch, "W": torch.randn(3, 3, requires_grad=True)}
        exec(
            "def helper(t):\n    return torch.relu(t @ W)\n\n"
            "def f(x):\n    return helper(x) * 2\n",
            ns,
        )
        f, W = ns["f"], ns["W"]
        x = torch.randn(3, 3)
        self.elide(f, x)
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(3):
                W.grad = None
                torch._rust.call_with_cfg_execution(f, x).sum().backward()
                got, W.grad = W.grad, None
                f(x).sum().backward()
                self.assertEqual(got, W.grad)
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
            training = torch._rust.cfg_observed(f, False, False).split("Autograd")[0]
            self.assertIn("guard W is tensor(f32 cpu [3, 3] stride [3, 1])", training)
            # A W of another shape fails the training version's check: the
            # plain version runs (no miss).
            ns["W"] = torch.randn(3, 2, requires_grad=True)
            misses = torch._rust._cfg_misses(f)[1]
            self.assertEqual(torch._rust.call_with_cfg_execution(f, x), f(x))
            self.assertEqual(torch._rust._cfg_misses(f)[1], misses)
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_elide_miss_keeps_params(self):
        # An elided version that never reads a param (here self, bound but
        # unused) still runs its origin on it when a check fails.
        class T:
            def m(self):
                return 1

        class U:
            def m(self):
                return 2

        def f(t):
            return t.m()

        self.elide(f, T())
        self.assertIn("guard s0 is ", torch._rust.cfg_observed(f, False, False))
        self.assertEqual(torch._rust.call_with_cfg_execution(f, U()), 2)
        self.assertEqual(torch._rust._cfg_misses(f)[1], 1)

    def test_specialize(self):
        # A region whose branch went one way becomes one block running the
        # path taken, with a check on the branch's condition in its guard
        # prefix (the local a can be read before foo runs); when the check
        # fails, the block's origin runs the region from its start.
        log = []
        ns = {"log": log}
        exec(
            "def foo(): log.append('foo')\n"
            "def bar(): log.append('bar')\n"
            "def baz(): log.append('baz')\n"
            "def bay(): log.append('bay')\n"
            "def f(a):\n    foo()\n    if a:\n        bar()\n    else:\n        baz()\n    bay()\n",
            ns,
        )
        f = ns["f"]
        torch._rust.call_with_cfg_execution(f, True)
        torch._rust._specialize(f, 0)
        self.assertExpectedInline(
            torch._rust.cfg_observed(f, False, False).split("origin:")[0],
            """\
bb0:
  version:
    guard type(a) is bool
    guard bool(a)
    Resume
    %0 = LoadGlobal foo
    %1 = Call 0 null, %0
    %2 = LoadGlobal bar
    %3 = Call 0 null, %2
    %4 = LoadGlobal bay
    %5 = Call 0 null, %4
    %6 = Const None
    outputs [%6]
  return o0
  """,
        )
        for a, want in (
            (True, ["foo", "bar", "bay"]),
            (False, ["foo", "baz", "bay"]),
            (0, ["foo", "baz", "bay"]),
        ):
            log.clear()
            torch._rust.call_with_cfg_execution(f, a)
            self.assertEqual(log, want)
        self.assertEqual(torch._rust._cfg_misses(f), [2])

    def test_specialize_compare(self):
        # A compare of values always seen exact builtins (each checked) runs
        # no Python code, so it moves ahead of foo into the guard.
        log = []
        ns = {"log": log}
        exec(
            "def foo(): log.append('foo')\n"
            "def bar(): log.append('bar')\n"
            "def baz(): log.append('baz')\n"
            "def f(n, s):\n    foo()\n    if n == 3 and s != 'x':\n        bar()\n    else:\n        baz()\n",
            ns,
        )
        f = ns["f"]
        torch._rust.call_with_cfg_execution(f, 3, "y")
        torch._rust._specialize(f, 0)
        out = torch._rust.cfg_observed(f, False, False).split("origin:")[0]
        for check in ("type(n) is int", "n == 3", "type(s) is str", "s != 'x'"):
            self.assertIn(f"guard {check}\n", out)
        self.assertNotIn("CompareOp", out)
        for args, want in (
            ((3, "y"), ["foo", "bar"]),
            ((4, "y"), ["foo", "baz"]),
            ((3, "x"), ["foo", "baz"]),
            ((3.0, "y"), ["foo", "bar"]),
        ):
            log.clear()
            torch._rust.call_with_cfg_execution(f, *args)
            self.assertEqual(log, want)
        self.assertEqual(torch._rust._cfg_misses(f), [3])

    def test_specialize_shapes(self):
        # Nested branches joining again, a value passed on the stack across
        # the branch, and an arm that raises (with CPython's traceback).
        def g(a, b, n):
            x = n * 2
            if a:
                x += 1
                if b:
                    x *= 10
                else:
                    x -= 3
            else:
                x = -x
            return x + 100

        def k(a, t):
            return [t, len("ab")] + [(t * 2 if a else t - 2)]

        def h(a, t):
            s = t + 1
            if not a:
                raise ValueError(f"bad {s}")
            return s * 2

        for fn, warm, cases in (
            (g, (True, True, 1), [(True, True, 1), (True, False, 2), (False, True, 3)]),
            (k, (True, 5), [(True, 5), (False, 5)]),
            (h, (True, 1), [(True, 1)]),
        ):
            torch._rust.call_with_cfg_execution(fn, *warm)
            torch._rust._specialize(fn, 0)
            shown = torch._rust.cfg_observed(fn, False, False)
            self.assertIn("guard ", shown)
            for args in cases:
                self.assertEqual(
                    torch._rust.call_with_cfg_execution(fn, *args), fn(*args)
                )
        self.assertEqual(torch._rust._cfg_misses(g)[0], 2)
        self.assertEqual(torch._rust._cfg_misses(k)[0], 1)
        try:
            torch._rust.call_with_cfg_execution(h, False, 2)
            self.fail("no exception")
        except ValueError as e:
            self.assertEqual(str(e), "bad 3")
            tb = traceback.extract_tb(e.__traceback__)
        self.assertEqual(tb[-1].line, 'raise ValueError(f"bad {s}")')

    def test_specialize_refused(self):
        # A branch seen going both ways isn't specialized, nor one whose
        # condition is only known after code that can run Python.
        def f(a):
            if a:
                return 1
            return 2

        def g(t):
            if t.ready():
                return 1
            return 2

        class T:
            def ready(self):
                return True

        torch._rust.call_with_cfg_execution(f, True)
        torch._rust.call_with_cfg_execution(f, False)
        torch._rust.call_with_cfg_execution(g, T())
        for fn in (f, g):
            with self.assertRaisesRegex(ValueError, "seen going one way"):
                torch._rust._specialize(fn, 0)

    def test_elide_stores(self):
        # Attribute stores are elided like the rest; a raise after them leaves
        # them done, as CPython does.
        class Counter:
            def __init__(self):
                self.n = 0

        def bump(c, t):
            c.n = c.n + 1
            c.last = t * 2
            return t + 1

        def f(c, x, z):
            return bump(c, x) + z

        x, z = torch.randn(3), torch.randn(3)
        self.elide(f, Counter(), x, z)
        version = torch._rust.cfg_observed(f, False, False).split("origin:")[0]
        self.assertIn("StoreAttr last", version)
        c = Counter()
        self.assertEqual(
            torch._rust.call_with_cfg_execution(f, c, x, z), f(Counter(), x, z)
        )
        self.assertEqual((c.n, c.last), (1, x * 2))
        c = Counter()
        with self.assertRaisesRegex(RuntimeError, "must match"):
            torch._rust.call_with_cfg_execution(f, c, x, torch.randn(2))
        self.assertEqual((c.n, c.last), (1, x * 2))

        class Slotted(Counter):
            __slots__ = ("last",)

        c = Slotted()
        self.assertEqual(
            torch._rust.call_with_cfg_execution(f, c, x, z), f(Slotted(), x, z)
        )
        self.assertEqual((c.n, c.last), (1, x * 2))
        self.assertEqual(torch._rust._cfg_misses(f)[1], 0)

    def test_elide_module(self):
        # A module call is elided through _wrapped_call_impl, _call_impl,
        # forward and __getattr__ (clones of each, optimized again once they
        # observed), and trains; a hook added later, or an instance forward,
        # fails a check and the real call runs.
        lin = torch.nn.Linear(4, 3)

        def f(x):
            return lin(x) * 2

        x = torch.randn(2, 4)
        torch._rust._set_cfg_optimize(True)
        torch._rust._set_cfg_elide(True)
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(15):
                lin.zero_grad()
                torch._rust.call_with_cfg_execution(f, x).sum().backward()
                got = [p.grad for p in lin.parameters()]
                lin.zero_grad()
                f(x).sum().backward()
                self.assertEqual(got, [p.grad for p in lin.parameters()])
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
            out = torch._rust.cfg_observed(f, False, False).split("origin:")[0]
            self.assertIn(
                "guard type(s0).__call__ is torch.nn.modules.module.Module._wrapped_call_impl",
                out,
            )
            self.assertIn(
                ".weight falls back to torch.nn.modules.module.Module.__getattr__", out
            )
            self.assertIn("Autograd", out)
            seen = []
            h = lin.register_forward_hook(lambda m, i, o: seen.append(1))
            self.assertEqual(torch._rust.call_with_cfg_execution(f, x), f(x))
            self.assertEqual(seen, [1, 1])
            misses = torch._rust._cfg_misses(f)[1]
            h.remove()
            lin.forward = lambda t: t[:, :3]
            self.assertEqual(torch._rust.call_with_cfg_execution(f, x), x[:, :3] * 2)
            self.assertEqual(torch._rust._cfg_misses(f)[1], misses + 1)
        finally:
            torch._rust._set_cfg_optimize(False)
            torch._rust._set_cfg_elide(False)
            torch._rust._set_cfg_autograd(False)

    def test_elide_config_branch(self):
        # A branch on Python config (a compare of attribute reads) is checked
        # in the guard, so the module call is elided and trains; when
        # the config changes, the check fails and the real call runs.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(4, 4)
                self.act = "relu"

            def forward(self, x):
                y = self.fc(x)
                if self.act == "relu":
                    return F.relu(y)
                return F.gelu(y)

        m = M()

        def f(x):
            return m(x)

        x = torch.randn(2, 4)
        torch._rust._set_cfg_optimize(True)
        torch._rust._set_cfg_elide(True)
        torch._rust._set_cfg_autograd(True)
        try:
            for _ in range(15):
                torch._rust.call_with_cfg_execution(f, x).sum().backward()
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
            out = torch._rust.cfg_observed(f, False, False)
            self.assertIn(".act == 'relu'\n", out)
            self.assertNotIn("CompareOp", out)
            misses = torch._rust._cfg_misses(f)[1]
            m.act = "gelu"
            self.assertEqual(torch._rust.call_with_cfg_execution(f, x), F.gelu(m.fc(x)))
            self.assertEqual(torch._rust._cfg_misses(f)[1], misses + 1)
        finally:
            torch._rust._set_cfg_optimize(False)
            torch._rust._set_cfg_elide(False)
            torch._rust._set_cfg_autograd(False)

    def test_callee_via_type(self):
        # Calling an instance records its class's __call__, and an attribute
        # load from an instance records its class's __getattr__ (entered,
        # directly, only when the generic lookup missed).
        class C:
            def __call__(self, x):
                return x + 1

            def __getattr__(self, name):
                return 2

        def f(c):
            return c(c.missing) + c.present

        c = C()
        c.present = 3
        self.same(f, c)
        out = re.sub(r"\S*<locals>\.", "", torch._rust.cfg_observed(f))
        self.assertExpectedInline(
            out,
            """\
bb0:
  Resume
  %0: C = LoadLocal c
  %1: C = LoadLocal c
  %2: int 2 = LoadAttr missing %1
      callee C.__getattr__ (via __getattr__)
      effects [1 visit]:
        call C.__getattr__
  %3: int 3 = Call 1 null, %0, %2
      callee C.__call__ (via __call__)
      effects [1 visit]:
        call C.__call__
  %4: C = LoadLocal c
  %5: int 3 = LoadAttr present %4
      callee C.__getattr__ (via __getattr__)
  %6: int 6 = BinaryOp + %3, %5
  return %6
""",
        )

    def test_autograd_int_tuple_param(self):
        # A tuple of ints a training version bakes in is guarded by value.
        def f(x, shape):
            return torch.mul(torch.reshape(x, shape), 2)

        x = torch.randn(2, 3)
        self.same(f, x, (3, 2))
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:
            for shape in [(3, 2), (3, 2), (3, 2), (6,), [3, 2]]:
                a = x.clone().requires_grad_()
                out = torch._rust.call_with_cfg_execution(f, a, shape)
                self.assertEqual(out, x.reshape(shape) * 2)
                out.sum().backward()
                self.assertEqual(a.grad, torch.full((2, 3), 2.0))
            self.assertEqual(torch._rust._cfg_autograd_blocks(f), [(1, 0, "trained")])
            self.assertIn("guard s1 == (3, 2)", torch._rust.cfg_observed(f))
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_autograd_traced_run_raises(self):
        # When the bootstrap's traced run raises (here a shape mismatch; a
        # TorchDispatchMode can also raise where eager wouldn't), the block
        # gives up and runs as usual, raising as eager does.
        def f(x, w):
            return x @ w * 2

        self.same(f, torch.randn(3, 3), torch.randn(3, 3))
        self.optimize_all(f)
        torch._rust._set_cfg_autograd(True)
        try:
            a, b = torch.randn(3, 2, requires_grad=True), torch.randn(3, 3)
            with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
                torch._rust.call_with_cfg_execution(f, a, b)
            (status,) = torch._rust._cfg_autograd_blocks(f)
            self.assertRegex(
                status[2],
                "^gave up: the traced run raised RuntimeError: .*cannot be multiplied",
            )
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_raise_repeats_no_mutation(self):
        # Nothing runs a pure block again after a raise: an in-place op
        # before it is seen once, version counter too.
        def f(x, w):
            x += 1
            return x @ w * 2

        torch._rust.call_with_cfg_execution(f, torch.randn(3, 3), torch.randn(3, 3))
        self.optimize_all(f)
        self.assertIn("aten::add_.Tensor", torch._rust.cfg_observed(f, False, False))
        torch._rust._set_cfg_autograd(True)
        try:
            x, w = torch.zeros(3, 2), torch.randn(3, 3, requires_grad=True)
            with self.assertRaisesRegex(RuntimeError, "cannot be multiplied"):
                torch._rust.call_with_cfg_execution(f, x, w)
            self.assertEqual(x, torch.ones(3, 2))
            self.assertEqual(x._version, 1)
        finally:
            torch._rust._set_cfg_autograd(False)

    def test_raise_repeats_no_hidden_writes(self):
        # batch_norm's running stats (written by native_batch_norm inside it)
        # are updated once when the block raises after it.
        def f(x, rm, rv, z):
            return torch.batch_norm(x, None, None, rm, rv, True, 0.1, 1e-5, False) + z

        x, z = torch.randn(4, 3), torch.randn(4, 3)
        torch._rust.call_with_cfg_execution(f, x, torch.zeros(3), torch.ones(3), z)
        self.optimize_all(f)
        stats = [(torch.zeros(3), torch.ones(3)) for _ in range(2)]
        for (rm, rv), run in zip(
            stats, (torch._rust.call_with_cfg_execution, lambda g, *a: g(*a))
        ):
            with self.assertRaisesRegex(RuntimeError, "must match"):
                run(f, x, rm, rv, torch.randn(2, 2))
        self.assertEqual(stats[0], stats[1])

    def test_guard_read_after_write(self):
        # A callee loaded before anything writes is pinned, its load checked
        # in the guard; one loaded after a write (here deleting an item)
        # isn't, since that check couldn't run on entry, and its call stays
        # a call. The read after the write sees it.
        def f(lst, x):
            y = torch.relu(x)
            del lst[0]
            return torch.neg(y), lst[0]

        x = torch.randn(2)
        torch._rust.call_with_cfg_execution(f, [1, 2, 3], x)
        self.optimize_all(f)
        out = torch._rust.cfg_observed(f, False, False).split("origin:")[0]
        self.assertIn("guard torch.relu is", out)
        self.assertNotIn("guard torch.neg is", out)
        y, second = torch._rust.call_with_cfg_execution(f, [1, 2, 3], x)
        self.assertEqual((y, second), (torch.neg(torch.relu(x)), 2))

    def test_raise_repeats_no_rng(self):
        # A random op before a raise draws once, as CPython does.
        def f(z):
            return torch.rand(3) + z

        torch._rust.call_with_cfg_execution(f, torch.randn(3))
        self.optimize_all(f)
        out = torch._rust.cfg_observed(f, False, False)
        self.assertIn("Pinned torch._VariableFunctionsClass.rand", out)
        after = []
        for run in (torch._rust.call_with_cfg_execution, lambda g, *a: g(*a)):
            torch.manual_seed(0)
            with self.assertRaisesRegex(RuntimeError, "must match"):
                run(f, torch.randn(2))
            after.append(torch.rand(2))
        self.assertEqual(after[0], after[1])

    def test_promote_locals_barriers(self):
        # Stores stay pending across what Dynamo would capture (here
        # torch.relu and Tensor.neg) and reach the frame before anything else
        # (here locals()).
        def f(x):
            y = x + 1
            d = locals()
            y = y * 2
            return d["y"], y

        def g(x):
            y = x + 1
            y = torch.relu(y)
            return y.neg()

        code = f.__code__.replace(co_name="f")
        f = type(f)(code, f.__globals__)
        self.same(f, 1)
        self.optimize_all(f)
        self.assertEqual(self.same(f, 1), (2, 4))
        version = torch._rust.cfg_observed(f).split("  origin:")[0]
        self.assertExpectedInline(
            version,
            """\
bb0:
  %0: int 1 = LoadLocal x
  jump bb1(s0=%0)
bb1(%1 s0: int 1):
  version:
    %2 = input locals
    Resume
    %3: int 1 = Const 1
    %4: int 2 = BinaryOp + %1, %3
    outputs [%4, %2]
  jump bb2(s0=o0, s1=o1)
""",
        )
        self.same(g, torch.ones(2))
        self.optimize_all(g)
        self.same(g, torch.ones(2))
        # y is stored only by the landing blocks a raise goes through.
        text = torch._rust.cfg_observed(g, effects=False, types=False)
        self.assertExpectedInline(
            text,
            """\
bb0:
  %0 = LoadLocal x
  jump bb1(s0=%0)
bb1(%1 s0):
  version:
    guard no torch function mode
    guard s0 is a plain tensor
    guard torch.relu is torch._VariableFunctionsClass.relu
    %2 = Pinned torch._VariableFunctionsClass.relu
    Resume
    %3 = Const 1
    %4 = Aten aten::add.Tensor %1, %3
    %5 = Call 1 null, %2, %4
    %6 = LoadMethod neg %5
    %7 = Project 0 %6
    %8 = Project 1 %6
    %9 = Call 0 %7, %8
    outputs [%9]
    unwind [4..5) -> bb3(s0=%4)
    unwind [5..9) -> bb4(s0=%5)
  jump bb2(s0=o0)
  origin:
    bb0(%0 s0):
      Resume
      %1 = Const 1
      %2 = BinaryOp + %0, %1
      %3 = LoadGlobal torch
      %4 = LoadAttr relu %3
      %5 = Call 1 null, %4, %2
      %6 = LoadMethod neg %5
      %7 = Project 0 %6
      %8 = Project 1 %6
      %9 = Call 0 %7, %8
      leave(%9)
      unwind [3..6) -> outer 0(%2)
      unwind [6..10) -> outer 1(%5)
bb2(%10 s0):
  return %10
bb3(%11 s0, %12 lasti, %13 exc):
  StoreLocal y %11
  reraise %13, lasti=%12
bb4(%14 s0, %15 lasti, %16 exc):
  StoreLocal y %14
  reraise %16, lasti=%15
""",
        )

    def test_promote_locals_unwind(self):
        # The handler, and a traceback's frame, see the locals stored before
        # the raise.
        def f(x, d):
            try:
                y = x + 1
                y = d[y]
                y = y * 2
            except KeyError:
                return -y
            return y

        def g(x):
            y = x + 1
            y = y * 3
            return y // 0

        self.same(f, 1, {2: 5})
        self.optimize_all(f)
        self.assertEqual(self.same(f, 1, {2: 5}), 10)
        self.assertEqual(self.same(f, 1, {}), -2)
        with self.assertRaises(ZeroDivisionError):
            torch._rust.call_with_cfg_execution(g, 1)
        self.optimize_all(g)
        try:
            torch._rust.call_with_cfg_execution(g, 1)
            self.fail("expected ZeroDivisionError")
        except ZeroDivisionError as e:
            tb = e.__traceback__
        while tb.tb_next is not None:
            tb = tb.tb_next
        self.assertEqual(tb.tb_frame.f_locals, {"x": 1, "y": 6})

    def test_promote_locals_unbound_and_cells(self):
        def f(xs, flag):
            if flag:
                y = 1
            r = [y for y in xs]  # noqa: C416
            del r
            return y

        def g(x):
            y = x + 1
            del y
            y = 3
            return y

        def h(x):
            x = x + 1
            y = x * 2
            return (lambda: x + y)()  # noqa: PLC3002

        class O:  # noqa: E742
            def m(self, v):
                return v + 1

        # The method load is a barrier but its projections aren't: they must
        # stay with it.
        def k(o, x):
            return o.m(x * 2)

        for fn, args in [(f, ([4], True)), (g, (1,)), (h, (1,)), (k, (O(), 3))]:
            self.same(fn, *args)
            self.optimize_all(fn)
            self.same(fn, *args)
        with self.assertRaises(UnboundLocalError):
            torch._rust.call_with_cfg_execution(f, [4], False)

    def test_tracing_is_refused(self):
        def f():
            sys.settrace(lambda *a: None)
            try:
                return (lambda: 1)()  # noqa: PLC3002
            finally:
                sys.settrace(None)

        with self.assertRaisesRegex(RuntimeError, "settrace"):
            torch._rust.call_with_cfg_execution(f)
        sys.settrace(None)


if __name__ == "__main__":
    run_tests()
