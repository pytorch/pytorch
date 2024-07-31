# Owner(s): ["oncall: jit"]

import os
import sys
from collections import namedtuple
from typing import Dict, List, NamedTuple, Tuple

import torch
from torch.testing._internal.common_utils import IS_WINDOWS
from torch.testing._internal.jit_utils import JitTestCase, make_global

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )


class TestTyping(JitTestCase):
    def test_dict_in_not_in(self):
        def test_in_dict(x):
            # type: (Dict[str, int]) -> bool
            return "hi" in x

        self.checkScript(test_in_dict, ({"hi": 2, "bye": 3},))
        self.checkScript(test_in_dict, ({"bye": 3},))

        # Check evaluation order
        @torch.jit.script
        def a():
            print("a")
            return 3

        @torch.jit.script
        def b():
            print("b")
            return {3: 2, 4: 1}

        @torch.jit.script
        def fn():
            return a() in b()

        with self.capture_stdout() as captured:
            self.assertTrue(fn())
        if not IS_WINDOWS:
            # no stdout capturing on windows
            self.assertEqual(captured[0], "a\nb\n")

        def test_not_in_dict(a):
            # type: (Dict[str, int]) -> bool
            if "hello" not in a:
                return False
            else:
                return True

        self.checkScript(test_not_in_dict, ({"hello": 1, "world": 2},))
        self.checkScript(test_not_in_dict, ({"world": 2},))

        def test_dict_tensor_key(a, t):
            # type: (Dict[Tensor, int], Tensor) -> bool
            if t in a:
                return True
            else:
                return False

        inp1 = torch.tensor(3)
        inp2 = torch.tensor(5)
        dict_a = {inp1: 1, inp2: 3}
        self.checkScript(test_dict_tensor_key, (dict_a, torch.tensor(4)))
        self.checkScript(test_dict_tensor_key, (dict_a, torch.tensor(3)))
        self.checkScript(test_dict_tensor_key, (dict_a, inp1))
        self.checkScript(test_dict_tensor_key, (dict_a, inp2))

    def test_list_type_refinement_annotation_element_mismatch(self):
        def fn():
            l: List[int] = [1, 2, "foo", 3]
            return l

        with self.assertRaisesRegex(
            RuntimeError,
            "List type annotation"
            r" `List\[int\]` did not match the "
            "types of the given list elements",
        ):
            torch.jit.script(fn)

    def test_dict_type_refinement_annotation_key_mismatch(self):
        def fn():
            l1 = [1, 2, "foo", 3]
            l2 = ["foo", "bar", "baz", "qux"]
            d: Dict[int, str] = dict(zip(l1, l2))
            return d

        with self.assertRaisesRegex(
            RuntimeError,
            "Dicts may only "
            "contain homogeneous keys, but the "
            "type of the first generated key "
            r"was Union\[int, str\]",
        ):
            torch.jit.script(fn)

    def test_dict_type_refinement_annotation_value_mismatch(self):
        def fn():
            l1 = ["foo", "bar", "baz", "qux"]
            l2 = [1, 2, "foo", 3]
            d: Dict[str, int] = dict(zip(l1, l2))
            return d

        with self.assertRaisesRegex(
            RuntimeError,
            "Dict type annotation"
            r" `Dict\[str, int\]` did not match"
            " the type of an actual value type"
            r" `Union\[int, str\]`",
        ):
            torch.jit.script(fn)

    def test_dict_invalid_annotations(self):
        # Check for invalid value type annotation
        def wrong_value_type(dictionary: Dict[str, torch.jit.ScriptModule]):
            return

        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            torch.jit.script(wrong_value_type)

        # Check for invalid key type annotation
        def wrong_key_type(dictionary: Dict[torch.jit.ScriptModule, str]):
            return

        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            torch.jit.script(wrong_key_type)

        # Check for invalid key and value type annotation
        def wrong_key_value_type(
            dictionary: Dict[torch.jit.ScriptModule, torch.jit.ScriptModule]
        ):
            return

        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            torch.jit.script(wrong_key_value_type)

    def test_tuple_specialization(self):
        @torch.jit.script
        def f(t, s):
            # type: (Tuple[Tensor, Tuple[int, Tensor]], str) -> Tensor
            x, t2 = t
            _, y = t2
            return x + y

        t = (
            torch.randn(2, 2),
            (1, torch.randn(2, 2)),
        )
        f(t, "hi")
        graph = f.graph_for(t, "hi")
        input_types = list(next(graph.inputs()).type().elements())
        w = input_types[0]
        self.assertEqual(input_types[0].kind(), "TensorType")
        self.assertEqual(input_types[1].elements()[1].kind(), "TensorType")

    def test_tuple_io(self):
        def stuff(x):
            # type: (Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
            a, b = x
            return b, a

        a = (torch.rand(3), torch.rand(3))
        self.checkScript(stuff, (a,))

    def test_tuple_keyword(self):
        def bar():
            f = tuple((1, 2))  # noqa: C409
            return f

        self.checkScript(bar, ())

        def foo():
            return tuple(1, 2)

        self.checkScriptRaisesRegex(foo, (), Exception, "1 argument")

        def cant_infer_size():
            return tuple([1, 2, 3])  # noqa: C409

        with self.assertRaisesRegex(Exception, "cannot statically infer the expected"):
            torch.jit.script(cant_infer_size)

    def test_tuple_create_return(self):
        def stuff2(x):
            # type: (int) -> Tuple[Tensor, Tensor]
            a = (torch.ones(x), torch.zeros(x))
            return a

        self.checkScript(stuff2, (3,))

    def test_list_io(self):
        def stuff3(x):
            # type: (List[int]) -> Tuple[Tensor, List[int]]
            return torch.ones(x), x

        self.checkScript(stuff3, ([3, 2],))

    def test_bool_list_io(self):
        @torch.jit.script
        def stuff4(x):
            # type: (List[bool]) -> Tuple[List[bool], List[bool], List[List[bool]]]
            return x, [True, False], [[True]]

        li_1, li_2, li_3 = stuff4([True])
        li_3 = li_3[0]
        for li in [li_1, li_2, li_3]:
            self.assertTrue(type(li[0]) == bool)

    def test_nested_list(self):
        def foo(z):
            # type: (Tuple[int, List[List[int]]]) -> int
            x, y = z
            return y[0][1]

        self.checkScript(foo, ((1, [[1, 2], [3, 4]]),))

    def test_list_sum(self):
        def fn(x: List[int]) -> int:
            return sum(x)

        def fn1(x: List[float]):
            return sum(x)

        def fn2(x: List[bool]):
            return sum(x)

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn1, ([1.0, 2.0, 3.0],))
        self.checkScript(fn1, ([1, 2.8, 3],))
        self.checkScript(fn2, ([True, False, False],))
        self.checkScript(fn2, ([False, False, False],))
        self.checkScript(fn2, ([0, 1, 1, 0],))

    def test_list_unification(self):
        def fn():
            return [1, None, 2]

        def fn2(x):
            return [torch.ones(2, 2), None, x]

        self.checkScript(fn, [])
        self.checkScript(fn2, (torch.ones(2, 2),))

    # to avoid defining sum_list in multiple tests
    def get_sum_list_fn(self):
        def sum_list(a):
            # type: (List[int]) -> int
            sum = 0
            for i in a:
                sum += i

            return sum

        return sum_list

    def test_sum_list_diff_elms(self):
        self.checkScript(self.get_sum_list_fn(), ([1, 2, 3, 4, 5],))

    def test_sum_list_empty(self):
        self.checkScript(self.get_sum_list_fn(), ([],))

    def test_sum_list_one(self):
        self.checkScript(self.get_sum_list_fn(), ([1],))

    def test_sum_list_literal(self):
        def sum_list():
            # type: () -> int
            sum = 0
            for i in [1, 2, 3, 4, 5]:
                sum += i

            return sum

        self.checkScript(sum_list, ())

    def test_sum_list_wrong_type(self):
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):

            @torch.jit.script
            def sum_list(a):
                # type: (int) -> int
                sum = 0
                for i in a:  # noqa: T484
                    sum += i

                return sum

            sum_list(1)

    def test_list_iterables(self):
        with self.assertRaisesRegex(
            RuntimeError, "List of iterables is not supported currently"
        ):
            cu = torch.jit.CompilationUnit(
                """
            def list_iterables(x):
                for i, j in [2, 3, 4], [5, 6, 7]:
                    x += i
                    x += j
                return x
            """
            )

    def test_for_in_string(self):
        def test_strings(x):
            # type: (str) -> str
            reverse = ""
            for c in x:
                reverse = c + reverse
            return reverse

        self.checkScript(test_strings, ("hello",))
        self.checkScript(test_strings, ("",))

        def test_list_strings(x):
            # type: (List[str]) -> str
            result = ""
            for sub_str in x:
                result += sub_str
            return result

        self.checkScript(test_list_strings, (["hello", "world"],))
        self.checkScript(test_list_strings, (["hello", " ", "world", ""],))

    def test_for_in_dict(self):
        def test_dicts(x):
            # type: (Dict[str, int]) -> int
            sum = 0
            for key in x:
                sum += x[key]
            return sum

        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

        def test_dict_keys_values(x):
            # type: (Dict[str, int]) -> Tuple[str, int]
            key_str = ""
            sum = 0
            for key in x.keys():
                key_str += key
            for val in x.values():
                sum += val
            return key_str, sum

        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

    def test_for_tuple_unpack(self):
        def for_tuple_unpack(x, y):
            for i, j in [[3, 4], [5, 6], [7, 8]]:
                x += i
                y += j
            return x, y

        self.checkScript(for_tuple_unpack, (torch.tensor(3), torch.tensor(5)))

        def nested_tuple_unpack(x, y):
            # type: (List[int], List[int]) -> int
            sum = 0
            for i, (j, k), v in zip(x, enumerate(x), y):
                sum += i + j + k + v
            return sum

        self.checkScript(nested_tuple_unpack, ([1, 3, 5], [2, 4, 6]))

    def test_dict_comprehension(self):
        def fn():
            return {i: chr(i + 65) for i in range(4)}

        self.checkScript(fn, ())

    def test_dict_comprehension_with_type_annotation(self):
        def fn():
            d: Dict[int, str] = {i: chr(i + 65) for i in range(4)}
            return d

        self.checkScript(fn, ())

        with self.assertRaisesRegex(RuntimeError, ""):
            with self.assertRaisesRegex(
                AssertionError,
                "Expected Dict "
                "type annotation for dict "
                "comprehension, found "
                "Tuple[int, str]",
            ):

                @torch.jit.script
                def fn():
                    d: Tuple[int, str] = {i: chr(i + 65) for i in range(4)}
                    return d

    def test_dict_comprehension_scope(self):
        def comprehension_can_access_outer_scope_variables():
            lst = ["foo", "bar", "baz"]
            return {l: len(l) for l in lst}

        self.checkScript(comprehension_can_access_outer_scope_variables, ())

        with self.assertRaisesRegex(RuntimeError, "undefined value i"):

            @torch.jit.script
            def outer_scope_cannot_access_comprehension_variables():
                d = {i: chr(i + 65) for i in range(4)}
                i = i + 1  # noqa: F821

    def test_for_tuple_assign(self):
        def test_simple_assign(x):
            # type: (Tuple[int, float]) -> float
            sum = 0.0
            for a in x:
                sum += float(a)
            return sum

        self.checkScript(test_simple_assign, ((1, 2.5),))

        def test_tuple_assign(x):
            # type: (Tuple[Tuple[int, int], Tuple[int, int]]) -> int
            sum = 0
            for a in x:
                sum += a[0]
                sum += a[1]
            return sum

        self.checkScript(test_tuple_assign, (((1, 2), (4, 7)),))

        def test_single_starred_lhs(self):
            with self.assertRaisesRegex(
                RuntimeError,
                "A Starred expression may only appear on the lhs within the presence"
                " of another non-starred expression",
            ):
                cu = torch.jit.CompilationUnit(
                    """
                def single_starred_lhs(x):
                    a = (x, x, x)
                    *b, = a
                    return b
                """
                )

    def test_singleton_tuple_unpack(self):
        def foo(a):
            (b,) = (a,)
            return b + 1

        self.checkScript(foo, (torch.rand(3),))

    def test_tuple_assignments(self):
        def var_tuple_assign(x, y):
            # type: (Tuple[Tensor, Tensor], Tensor) -> Tensor
            (a, b), c = x, y
            return a + b + c

        tuple_inputs = (torch.randn(1, 4), torch.randn(3, 4))
        self.checkScript(var_tuple_assign, (tuple_inputs, torch.randn(3, 4)))

        def nested_tuple_assign(x, y, z):
            # type: (int, Tuple[int, Tuple[int, int]], Tuple[int, int]) -> int
            a, (b, (c, d)), (e, f) = x, y, z
            return a + b + c + d + e + f

        self.checkScript(nested_tuple_assign, ((1, (2, (3, 4)), (5, 6))))

        def subscript_tuple_assign(a, x, i):
            # type: (List[int], Tensor, int) -> Tuple[int, Tensor, int]
            a[i], (x[i], b) = 1, (2, 3)
            return a[i] + 1, x + 5, b

        self.checkScript(
            subscript_tuple_assign, ([12, 7, 9, 11], torch.tensor((3, 13, 17)), 0)
        )

        def star_tuple_assign():
            # type: () -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]
            a, (b, *c), *d = 1, (2, 3, 4), 5, 6
            return a, b, c, d

        self.checkScript(star_tuple_assign, ())

        def subscript_tuple_augmented_assign(a):
            # type: (Tuple[int, int]) -> Tuple[int, int]
            a[0] += 1
            return a

        with self.assertRaisesRegex(RuntimeError, "does not support augmented assign"):
            scripted_aug_assign = torch.jit.script(subscript_tuple_augmented_assign)

    def test_multiple_assign(self):
        def test():
            a = b, c = d, f = (1, 1)

            # side effect
            ten = torch.tensor(1)
            ten1 = ten2 = ten.add_(1)

            # ordering
            x = 1
            y = 3
            x, y = y, x + y

            return a, b, c, d, f, ten, ten1, ten2, x, y

        self.checkScript(test, ())

    def test_opt_opt_refinement(self):
        @torch.jit.script
        def test_unify(weight, bias):
            # type: (Optional[int], Optional[int]) -> Optional[int]
            if weight is not None:
                opt = None
            else:
                if bias is not None:
                    opt = 1
                else:
                    opt = None

            return opt

    def test_optional_refinement(self):
        @torch.jit.script
        def test_if_none_assignment(x):
            # type: (Optional[int]) -> int
            if x is None:
                x = 1
            return x + 1

        self.assertEqual(test_if_none_assignment(1), 2)

    def test_optional_conversion(self):
        @torch.jit.script
        def other_fn(x=None):
            # type: (Optional[int]) -> int
            return torch.jit._unwrap_optional(x)

        @torch.jit.script
        def fn(x):
            # type: (int) -> int
            return other_fn(x)

        self.assertEqual(fn(2), 2)

        @torch.jit.script
        def unify_to_optional(x):
            # type: (bool) -> Optional[int]
            if x:
                a = None
            else:
                a = 2
            return a

        self.assertEqual(unify_to_optional(True), None)
        self.assertEqual(unify_to_optional(False), 2)

        @torch.jit.script
        def opt_list(x):
            # type: (Optional[List[float]]) -> int
            return 2

        @torch.jit.script
        def broadcast_opt_list(x):
            # type: (Optional[BroadcastingList2[float]]) -> int
            return 2

        @torch.jit.script
        def opt_list_tuple_caller(x):
            # type: (Tuple[float, float]) -> int
            return opt_list(x) + broadcast_opt_list(x)

        self.assertEqual(opt_list_tuple_caller((2.0, 3.0)), 4)

    def test_optional_tuple(self):
        def fn(x=None):
            # type: (Optional[Tuple[int, int]]) -> Tuple[int, int]
            if x is None:
                new_x = (1, 2)
            else:
                new_x = x
            return new_x

        self.checkScript(fn, ((3, 4),))
        self.checkScript(fn, ())

    def test_namedtuple_redefine(self):
        global _1, _2
        _1 = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
        _2 = namedtuple("GoogLeNetOutputs", ["different"])

        with self.assertRaisesRegex(RuntimeError, r"redefine"):

            @torch.jit.script
            def foo(x, y):
                # type: (_1, _2) -> _1
                return x

    def test_namedtuple_py2(self):
        global _GoogLeNetOutputs  # see [local resolution in python]
        _GoogLeNetOutputs = namedtuple(
            "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
        )

        @torch.jit.script
        def foo(x):
            # type: (_GoogLeNetOutputs) -> _GoogLeNetOutputs
            return x

        vals = torch.rand(3), torch.rand(4), torch.rand(5)
        out = foo(
            _GoogLeNetOutputs(logits=vals[0], aux_logits2=vals[1], aux_logits1=vals[2])
        )
        self.assertEqual(out.logits, vals[0])
        self.assertEqual(out.aux_logits2, vals[1])
        self.assertEqual(out.aux_logits1, vals[2])

    def test_namedtuple_good_error(self):
        global _GoogLeNetOutputs  # see [local resolution in python]
        _GoogLeNetOutputs = namedtuple(
            "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
        )

        @torch.jit.script
        def foo(x):
            # type: (_GoogLeNetOutputs) -> _GoogLeNetOutputs
            return x

        with self.assertRaisesRegex(
            RuntimeError, r"aka NamedTuple\(logits, aux_logits2, aux_logits1\)"
        ):
            out = foo(_GoogLeNetOutputs(logits="3", aux_logits2="4", aux_logits1="5"))

    def test_namedtuple_error_source_attribution(self):
        class _NamedTupleBadMemberType(NamedTuple):
            f1: torch.Tensor
            f2: "ABadForwardRefType"  # noqa: F821

        make_global(_NamedTupleBadMemberType)  # see [local resolution in python]

        def fn(x: _NamedTupleBadMemberType) -> torch.Tensor:
            return x.f1.relu()

        # assert that this has a location associated with the error.
        # note the " +" is regex (i.e. "at least one space")
        with self.assertRaisesRegex(ValueError, "at +File"):
            torch.jit.script(fn)

    def test_inherited_annotations_python_310(self):
        # See #104484
        # In python >=3.10, inspect.get_annotations doesn't always return the same values.
        # Sometimes it will show all annotations; other times it will show only annotations
        # that show in that class, not classes it inherits fro.
        class BaseModule(torch.nn.Module):
            state: List[int]

            def forward(self, x):
                pass

        def do_something_with_list(x: List[int]):
            if x:
                return x[-1]
            return 5

        class Submodule(BaseModule):
            def __init__(self, self_x_value):
                super().__init__()
                self.x = self_x_value
                self.state = []

            def forward(self, x):
                return self.x + x + do_something_with_list(self.state)

        class LowestModule(Submodule):
            def __init__(self) -> None:
                super().__init__(123)

        mod = LowestModule()
        mod2 = LowestModule()
        mod_s = torch.jit.script(mod)
        mod2_s = torch.jit.script(mod2)
