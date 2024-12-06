# Owner(s): ["module: dynamo"]

import collections
import dis
import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo import bytecode_analysis, bytecode_transformation
from torch._dynamo.testing import skipIfNotPy311, skipIfNotPy312


class BytecodeTests(torch._dynamo.test_case.TestCase):
    @skipIfNotPy311
    def test_linetable_311_writer1(self):
        def fn():
            a = 10
            b = 20
            # prevent LOAD_FAST_LOAD_FAST in 3.13 by wrapping b with g()
            c = a + g(b)
            f = "linetable_writer"
            return f"Test if {f} generates correct co_linetable: {c}"

        keys = bytecode_transformation.get_code_keys()
        code_options = {k: getattr(fn.__code__, k) for k in keys}
        result = bytecode_transformation.clean_and_assemble_instructions(
            bytecode_transformation.cleaned_instructions(fn.__code__),
            keys,
            code_options,
        )
        l1, l2 = list(fn.__code__.co_positions()), list(result[1].co_positions())
        self.assertEqual(len(l1), len(l2))
        for p1, p2 in zip(l1, l2):
            self.assertEqual(p1, p2)
        # TODO co_lnotab is deprecated in 3.12 and will be removed in 3.14
        # In 3.11+,. it is computed lazily from other linetable attributes (e.g. co_linetable),
        # so we do not set this attribute ourselves.
        self.assertEqual(fn.__code__.co_lnotab, result[1].co_lnotab)

    @skipIfNotPy311
    def test_linetable_311_writer2(self):
        """
        test large ops (LOAD_METHOD) and EXTENDED_ARGS
        fn_str is in the form:
        def fn():
            ...
            x0 = 1
            x1 = 1
            ...
            l = [x0, x1, ...]
        """
        fn_str = f"""\
def fn():
    foo.bar(1, 2, 3)
{str(chr(10)).join(' ' * 4 + 'x' + str(i) + ' = 1' for i in range(1 << 9))}
    l = [{' '.join('x' + str(i) + ',' for i in range(1 << 9))}]
        """
        locals = {}
        exec(fn_str, {}, locals)
        fn = locals["fn"]
        orig_inst_str = "\n".join(list(map(str, dis.get_instructions(fn))))
        self.assertIn("EXTENDED_ARG", orig_inst_str)
        load_method_str = "LOAD_ATTR" if sys.version_info >= (3, 12) else "LOAD_METHOD"
        self.assertIn(load_method_str, orig_inst_str)
        keys = bytecode_transformation.get_code_keys()
        code_options = {k: getattr(fn.__code__, k) for k in keys}
        result = bytecode_transformation.clean_and_assemble_instructions(
            bytecode_transformation.cleaned_instructions(fn.__code__),
            keys,
            code_options,
        )
        new_inst_str = "\n".join(list(map(str, result[0])))
        self.assertIn("EXTENDED_ARG", new_inst_str)
        self.assertIn(load_method_str, new_inst_str)
        l1, l2 = list(fn.__code__.co_positions()), list(result[1].co_positions())
        self.assertEqual(len(l1), len(l2))
        for p1, p2 in zip(l1, l2):
            self.assertEqual(p1, p2)
        self.assertEqual(fn.__code__.co_lnotab, result[1].co_lnotab)

    @unittest.skipIf(
        sys.version_info < (3, 10) or sys.version_info >= (3, 11),
        "linetable test for Python 3.10",
    )
    def test_linetable_310_writer(self):
        def fn():
            a = 10
            b = 20
            c = a + b
            f = "linetable_writer"
            return f"Test if {f} generates correct co_linetable: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_linetable)

    @unittest.skipIf(sys.version_info >= (3, 10), "use lnotab when python < 3.10")
    def test_lnotab_writer(self):
        def fn():
            a = 10
            b = 20
            c = a + b
            f = "lnotab_writer"
            return f"Test if {f} generates correct co_lnotab: {c}"

        inst = dis.get_instructions(fn)
        result = bytecode_transformation.assemble(inst, fn.__code__.co_firstlineno)
        self.assertTrue(result[1] == fn.__code__.co_lnotab)

    def test_if_tensor_is_none(self):
        """
        Python 3.11 adds new jump instructions that check if
        TOS is None. We do not support these instructions.
        """

        def f(x, y):
            z = 1
            if x is None:
                z *= 2
            if y is not None:
                z *= 3
            return z

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(None, torch.ones(2)), 6)

        if sys.version_info >= (3, 11):
            insts = bytecode_transformation.cleaned_instructions(f.__code__)
            for inst in insts:
                self.assertNotIn("_NONE", inst.opname)

    @skipIfNotPy311
    def test_py311_jump_offset(self):
        new_inst = bytecode_transformation.create_instruction
        consts = (None, 1, 2, 3, 4)

        def create_test_code(jump_opname, target_idx):
            targets = [
                new_inst("LOAD_CONST", argval=1),
                new_inst("LOAD_CONST", argval=3),
            ]
            jump_to_target_inst = new_inst(jump_opname, target=targets[target_idx])
            """
            pseudocode of generated bytecode:
            def test_py311_fn():
                goto target1
            target0:
                return 1
            target1:
                goto [target0/target2] (via fwd or bwd jump)
                return 2
            target2:
                return 3
                return 4
            """
            # test with LOAD_GLOBAL since it has a different instruction size
            insts = [
                new_inst("RESUME", arg=0),
                new_inst("JUMP_FORWARD", target=jump_to_target_inst),
                targets[0],
                new_inst("LOAD_GLOBAL", arg=0, argval="print"),
                new_inst("POP_TOP"),
                new_inst("RETURN_VALUE"),
                jump_to_target_inst,
                new_inst("LOAD_CONST", argval=2),
                new_inst("LOAD_GLOBAL", arg=0, argval="print"),
                new_inst("POP_TOP"),
                new_inst("RETURN_VALUE"),
                targets[1],
                new_inst("RETURN_VALUE"),
                new_inst("LOAD_CONST", argval=4),
                new_inst("RETURN_VALUE"),
            ]
            code_options = collections.OrderedDict(
                [
                    ("co_argcount", 0),
                    ("co_posonlyargcount", 0),
                    ("co_kwonlyargcount", 0),
                    ("co_nlocals", 0),
                    ("co_stacksize", 2),
                    ("co_flags", 3),
                    ("co_code", b""),
                    ("co_consts", consts),
                    ("co_names", ("print",)),
                    ("co_varnames", ()),
                    ("co_filename", __file__),
                    ("co_name", "test_py311_fn"),
                    ("co_qualname", "test_py311_fn"),
                    ("co_firstlineno", 1),
                    ("co_linetable", b""),
                    ("co_exceptiontable", b""),
                    ("co_freevars", ()),
                    ("co_cellvars", ()),
                ]
            )
            return bytecode_transformation.clean_and_assemble_instructions(
                insts,
                list(code_options.keys()),
                code_options,
            )

        # format: jump_opname, target_idx, expected forward jump, expected return value
        test_args = (
            ("JUMP_FORWARD", 0, False, 1),
            ("JUMP_FORWARD", 1, True, 3),
            ("JUMP_BACKWARD", 0, False, 1),
            ("JUMP_BACKWARD", 1, True, 3),
        )

        for test in test_args:
            insts, code = create_test_code(test[0], test[1])
            # check if offset of latest jump instruction is forward/backward
            for inst in reversed(insts):
                if inst.opname.startswith("JUMP"):
                    if test[2]:
                        self.assertIn("FORWARD", inst.opname)
                    else:
                        self.assertIn("BACKWARD", inst.opname)
                    break
            # run the code and check result

            def dummy_fn():
                pass

            dummy_fn.__code__ = code
            self.assertEqual(dummy_fn(), test[3])

            dummy_opt = torch.compile(dummy_fn, backend="eager")
            self.assertEqual(dummy_opt(), test[3])

    def test_exception_table_encode_varint(self):
        # these numbers have no real meaning to them
        nums = [
            0b111_101010_000000,
            0b1100_111000_010101_101010,
        ]
        b = bytecode_transformation.encode_exception_table_varint(
            nums[0]
        ) + bytecode_transformation.encode_exception_table_varint(nums[1])
        nums_new = []
        b_iter = iter(bytes(b))
        while True:
            try:
                nums_new.append(
                    bytecode_transformation.decode_exception_table_varint(b_iter)
                )
            except StopIteration:
                break
        self.assertEqual(nums, nums_new)

    @skipIfNotPy311
    def test_exception_table_parsing(self):
        def fn():
            try:
                with a():
                    b()
                c()
            except Exception:
                d()
            finally:
                e()
            f()

        tab = bytecode_transformation.parse_exception_table(
            fn.__code__.co_exceptiontable
        )
        b = bytecode_transformation.assemble_exception_table(tab)
        self.assertEqual(b, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_e2e(self):
        def fn():
            try:
                with a():
                    b()
                c()
            except Exception:
                d()
            finally:
                e()
            f()

        def nothing(*args):
            pass

        code = bytecode_transformation.transform_code_object(fn.__code__, nothing)
        self.assertEqual(code.co_exceptiontable, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_e2e_2(self):
        # last instructions of an exn_table entry is a large instruction
        # i.e., LOAD_GLOBAL a
        def fn():
            try:
                return a
            except Exception:
                pass

        def nothing(*args):
            pass

        code = bytecode_transformation.transform_code_object(fn.__code__, nothing)
        self.assertEqual(code.co_exceptiontable, fn.__code__.co_exceptiontable)

    @skipIfNotPy311
    def test_exception_table_entry_propagation(self):
        insts = []
        for _ in range(10):
            insts.append(bytecode_transformation.create_instruction("NOP"))
        insts[8].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[9], insts[0], 0, True
        )
        insts[0].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[0], insts[1], 0, True
        )
        insts[1].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[0], insts[2], insts[2], 0, True
        )
        insts[5].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[4], insts[6], insts[3], 0, True
        )
        insts[9].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[9], insts[9], insts[4], 0, True
        )
        insts[7].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[7], insts[9], insts[5], 0, True
        )
        bytecode_transformation.propagate_inst_exn_table_entries(insts)
        expected = [1, 2, 2, 0, 3, 3, 3, 5, 5, 4]
        for inst, exp in zip(insts, expected):
            self.assertIsNotNone(inst.exn_tab_entry)
            self.assertIs(inst.exn_tab_entry.target, insts[exp])

    @skipIfNotPy311
    def test_compute_exception_table_nested(self):
        insts = []
        for _ in range(20):
            insts.append(bytecode_transformation.create_instruction("NOP"))
        insts[10].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[10], insts[0], 0, True
        )
        insts[0].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[1], insts[1], 0, True
        )
        insts[1].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[1], insts[3], insts[2], 0, True
        )
        insts[5].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[5], insts[7], insts[3], 0, True
        )
        insts[9].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[10], insts[10], insts[4], 0, True
        )
        insts[7].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[8], insts[10], insts[5], 0, True
        )
        insts[14].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[13], insts[17], insts[6], 0, True
        )
        insts[16].exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            insts[15], insts[16], insts[7], 0, True
        )
        bytecode_transformation.update_offsets(insts)
        tab = bytecode_transformation.compute_exception_table(insts)
        expected = [
            (1, 1, 1),
            (2, 3, 2),
            (4, 4, 0),
            (5, 7, 3),
            (8, 9, 5),
            (10, 10, 4),
            (13, 14, 6),
            (15, 16, 7),
            (17, 17, 6),
        ]
        self.assertEqual(len(tab), len(expected))
        for entry, exp in zip(tab, expected):
            self.assertEqual(entry.start, exp[0] * 2)
            self.assertEqual(entry.end, exp[1] * 2)
            self.assertEqual(entry.target, exp[2] * 2)

    @skipIfNotPy311
    def test_remove_dead_code_with_exn_table_entries(self):
        create_instruction = bytecode_transformation.create_instruction
        target1 = create_instruction("NOP")
        target2 = create_instruction("NOP")
        target3 = create_instruction("NOP")
        exn_start = create_instruction("NOP")
        exn_end = create_instruction("NOP")
        insts = [
            create_instruction("JUMP_FORWARD", target=target1),
            exn_start,  # dead
            target1,
            create_instruction("JUMP_FORWARD", target=target3),
            exn_end,  # dead
            target2,
            target3,
        ]
        exn_start.exn_tab_entry = bytecode_transformation.InstructionExnTabEntry(
            exn_start, exn_end, target2, 0, True
        )
        bytecode_transformation.propagate_inst_exn_table_entries(insts)
        insts = bytecode_analysis.remove_dead_code(insts)
        self.assertEqual(len(insts), 5)
        self.assertNotIn(exn_start, insts)
        self.assertNotIn(exn_end, insts)
        self.assertIn(target2, insts)
        self.assertIn(target3, insts)
        bytecode_transformation.update_offsets(insts)
        tab = bytecode_transformation.compute_exception_table(insts)
        self.assertEqual(len(tab), 1)
        self.assertEqual(tab[0].start, 2)
        self.assertEqual(tab[0].end, 4)
        self.assertEqual(tab[0].target, 6)

    def test_bytecode_from_template(self):
        def fn(d1):
            for k, v in d1.items():
                d2[k] = v

        varname_map = {"d1": "var1", "d2": "var2", "k": "var3", "v": "var4"}
        insts = bytecode_transformation.bytecode_from_template(fn, varname_map)
        for inst in insts:
            self.assertIsNone(inst.starts_line)
            if inst.opname.startswith("LOAD"):
                self.assertNotIn(inst.argval, varname_map)
                if inst.opname not in ("LOAD_GLOBAL", "LOAD_ATTR"):
                    self.assertIsNone(inst.arg)
            self.assertFalse(inst.opname.startswith("RETURN"))

    @skipIfNotPy311
    def test_bytecode_from_template_noprefix(self):
        # Test that 3.11+ prefix instructions are removed
        def gen_fn():
            cl = None

            def fn():
                return cl

            return fn

        fn = gen_fn()

        dis_insts = list(dis.get_instructions(fn))
        names = {inst.opname for inst in dis_insts}
        self.assertIn("RESUME", names)
        self.assertIn("COPY_FREE_VARS", names)

        insts = bytecode_transformation.bytecode_from_template(fn)
        names = {inst.opname for inst in insts}
        self.assertNotIn("RESUME", names)
        self.assertNotIn("COPY_FREE_VARS", names)

    def test_bytecode_from_template_noreturn1(self):
        # Test that functions with multiple returns will have their
        # returns replaced with jumps to the end
        def fn():
            if x:
                return y
            z = 3
            return z

        dis_insts = list(dis.get_instructions(fn))
        dis_returns = list(filter(lambda x: x.opname.startswith("RETURN"), dis_insts))
        self.assertGreater(len(dis_returns), 1)
        self.assertTrue(dis_insts[-1].opname.startswith("RETURN"))

        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        self.assertEqual(insts[-1].opname, "NOP")
        self.assertEqual(len(dis_insts), len(insts))
        for i0, i1 in zip(dis_insts, insts):
            if i0.opname.startswith("RETURN"):
                if i1 is insts[-1]:
                    continue
                self.assertIn("JUMP", i1.opname)
                self.assertIs(i1.target, insts[-1])

    # Should work with 3.10, but testing with 3.11+ is sufficient.
    # In 3.8, `fn` ends with a RETURN_VALUE.
    @skipIfNotPy311
    def test_bytecode_from_template_noreturn2(self):
        # Test function that doesn't end with RETURN_VALUE
        def fn():
            if x:
                return x
            if x:
                return x
            raise RuntimeError

        dis_insts = list(dis.get_instructions(fn))
        self.assertFalse(dis_insts[-1].opname.startswith("RETURN"))

        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        self.assertEqual(insts[-1].opname, "NOP")
        self.assertEqual(insts[-2].opname, dis_insts[-1].opname)
        self.assertEqual(len(dis_insts) + 1, len(insts))
        for i0, i1 in zip(dis_insts, insts):
            if i0.opname.startswith("RETURN"):
                self.assertIn("JUMP", i1.opname)
                self.assertIs(i1.target, insts[-1])

    @skipIfNotPy312
    def test_bytecode_from_template_noreturn_const(self):
        # Test 3.12+ RETURN_CONST
        def fn():
            if x:
                return 1
            return 0

        dis_insts = list(dis.get_instructions(fn))
        dis_return_consts = list(
            filter(lambda x: x.opname == "RETURN_CONST", dis_insts)
        )
        self.assertGreater(len(dis_return_consts), 1)
        self.assertTrue(dis_insts[-1].opname == "RETURN_CONST")

        insts = bytecode_transformation.bytecode_from_template(fn, noprefix=False)
        self.assertEqual(insts[-1].opname, "NOP")
        insts_i = 0
        for i, inst in enumerate(dis_insts):
            if inst.opname == "RETURN_CONST":
                self.assertEqual(insts[insts_i].opname, "LOAD_CONST")
                insts_i += 1
                if insts_i != len(insts) - 1:
                    self.assertIn("JUMP", insts[insts_i].opname)
                    self.assertIs(insts[insts_i].target, insts[-1])
            insts_i += 1

    def test_bytecode_analysis_jump_backward_no_interrupt(self):
        # bytecode_analysis fails if JUMP_BACKWARD_NO_INTERRUPT is not terminal in 3.13+
        @torch.compile(backend="eager")
        def fn(x):
            # graph break causes bytecode_analysis to analyze the rest of this function
            torch._dynamo.graph_break()
            with torch.no_grad():
                try:
                    x = x + 1
                except NotImplementedError:
                    x = x + 1
                except Exception as e:
                    x = x + 1
            return x

        self.assertEqual(fn(torch.ones(3)), torch.ones(3) + 1)


class BytecodeHookTests(torch._dynamo.test_case.TestCase):
    def test_bytecode_hook(self):
        def fn(a, b):
            return a - b * 10

        def hook(code, out_code):
            print(code)
            print(out_code)
            return code

        torch._dynamo.reset()
        handle = torch._dynamo.convert_frame.register_bytecode_hook(hook)
        try:
            opt_fn = torch.compile(fn)
            for i in range(2, 12):
                opt_fn(torch.randn(i), torch.randn(i))
        finally:
            handle.remove()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
