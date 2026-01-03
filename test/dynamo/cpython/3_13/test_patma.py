# Owner(s): ["module: dynamo"]
import array
import collections
import dataclasses
import dis
import enum
import inspect
import sys
import unittest
from test import support


@dataclasses.dataclass
class Point:
    x: int
    y: int


class TestCompiler(unittest.TestCase):

    def test_refleaks(self):
        # Hunting for leaks using -R doesn't catch leaks in the compiler itself,
        # just the code under test. This test ensures that if there are leaks in
        # the pattern compiler, those runs will fail:
        with open(__file__) as file:
            compile(file.read(), __file__, "exec")


class TestInheritance(unittest.TestCase):

    @staticmethod
    def check_sequence_then_mapping(x):
        match x:
            case [*_]:
                return "seq"
            case {}:
                return "map"

    @staticmethod
    def check_mapping_then_sequence(x):
        match x:
            case {}:
                return "map"
            case [*_]:
                return "seq"

    def test_multiple_inheritance_mapping(self):
        class C:
            pass
        class M1(collections.UserDict, collections.abc.Sequence):
            pass
        class M2(C, collections.UserDict, collections.abc.Sequence):
            pass
        class M3(collections.UserDict, C, list):
            pass
        class M4(dict, collections.abc.Sequence, C):
            pass
        self.assertEqual(self.check_sequence_then_mapping(M1()), "map")
        self.assertEqual(self.check_sequence_then_mapping(M2()), "map")
        self.assertEqual(self.check_sequence_then_mapping(M3()), "map")
        self.assertEqual(self.check_sequence_then_mapping(M4()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M1()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M2()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M3()), "map")
        self.assertEqual(self.check_mapping_then_sequence(M4()), "map")

    def test_multiple_inheritance_sequence(self):
        class C:
            pass
        class S1(collections.UserList, collections.abc.Mapping):
            pass
        class S2(C, collections.UserList, collections.abc.Mapping):
            pass
        class S3(list, C, collections.abc.Mapping):
            pass
        class S4(collections.UserList, dict, C):
            pass
        self.assertEqual(self.check_sequence_then_mapping(S1()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(S2()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(S3()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(S4()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S1()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S2()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S3()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(S4()), "seq")

    def test_late_registration_mapping(self):
        class Parent:
            pass
        class ChildPre(Parent):
            pass
        class GrandchildPre(ChildPre):
            pass
        collections.abc.Mapping.register(Parent)
        class ChildPost(Parent):
            pass
        class GrandchildPost(ChildPost):
            pass
        self.assertEqual(self.check_sequence_then_mapping(Parent()), "map")
        self.assertEqual(self.check_sequence_then_mapping(ChildPre()), "map")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPre()), "map")
        self.assertEqual(self.check_sequence_then_mapping(ChildPost()), "map")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPost()), "map")
        self.assertEqual(self.check_mapping_then_sequence(Parent()), "map")
        self.assertEqual(self.check_mapping_then_sequence(ChildPre()), "map")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPre()), "map")
        self.assertEqual(self.check_mapping_then_sequence(ChildPost()), "map")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPost()), "map")

    def test_late_registration_sequence(self):
        class Parent:
            pass
        class ChildPre(Parent):
            pass
        class GrandchildPre(ChildPre):
            pass
        collections.abc.Sequence.register(Parent)
        class ChildPost(Parent):
            pass
        class GrandchildPost(ChildPost):
            pass
        self.assertEqual(self.check_sequence_then_mapping(Parent()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(ChildPre()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPre()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(ChildPost()), "seq")
        self.assertEqual(self.check_sequence_then_mapping(GrandchildPost()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(Parent()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(ChildPre()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPre()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(ChildPost()), "seq")
        self.assertEqual(self.check_mapping_then_sequence(GrandchildPost()), "seq")


class TestPatma(unittest.TestCase):

    def test_patma_000(self):
        match 0:
            case 0:
                x = True
        self.assertIs(x, True)

    def test_patma_001(self):
        match 0:
            case 0 if False:
                x = False
            case 0 if True:
                x = True
        self.assertIs(x, True)

    def test_patma_002(self):
        match 0:
            case 0:
                x = True
            case 0:
                x = False
        self.assertIs(x, True)

    def test_patma_003(self):
        x = False
        match 0:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_004(self):
        x = False
        match 1:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_005(self):
        x = False
        match 2:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_006(self):
        x = False
        match 3:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, True)

    def test_patma_007(self):
        x = False
        match 4:
            case 0 | 1 | 2 | 3:
                x = True
        self.assertIs(x, False)

    def test_patma_008(self):
        x = 0
        class A:
            y = 1
        match x:
            case A.y as z:
                pass
        self.assertEqual(x, 0)
        self.assertEqual(A.y, 1)

    def test_patma_009(self):
        class A:
            B = 0
        match 0:
            case x if x:
                z = 0
            case _ as y if y == x and y:
                z = 1
            case A.B:
                z = 2
        self.assertEqual(A.B, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 2)

    def test_patma_010(self):
        match ():
            case []:
                x = 0
        self.assertEqual(x, 0)

    def test_patma_011(self):
        match (0, 1, 2):
            case [*x]:
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_012(self):
        match (0, 1, 2):
            case [0, *x]:
                y = 0
        self.assertEqual(x, [1, 2])
        self.assertEqual(y, 0)

    def test_patma_013(self):
        match (0, 1, 2):
            case [0, 1, *x,]:
                y = 0
        self.assertEqual(x, [2])
        self.assertEqual(y, 0)

    def test_patma_014(self):
        match (0, 1, 2):
            case [0, 1, 2, *x]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_015(self):
        match (0, 1, 2):
            case [*x, 2,]:
                y = 0
        self.assertEqual(x, [0, 1])
        self.assertEqual(y, 0)

    def test_patma_016(self):
        match (0, 1, 2):
            case [*x, 1, 2]:
                y = 0
        self.assertEqual(x, [0])
        self.assertEqual(y, 0)

    def test_patma_017(self):
        match (0, 1, 2):
            case [*x, 0, 1, 2,]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_018(self):
        match (0, 1, 2):
            case [0, *x, 2]:
                y = 0
        self.assertEqual(x, [1])
        self.assertEqual(y, 0)

    def test_patma_019(self):
        match (0, 1, 2):
            case [0, 1, *x, 2,]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_020(self):
        match (0, 1, 2):
            case [0, *x, 1, 2]:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_021(self):
        match (0, 1, 2):
            case [*x,]:
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_022(self):
        x = {}
        match x:
            case {}:
                y = 0
        self.assertEqual(x, {})
        self.assertEqual(y, 0)

    def test_patma_023(self):
        x = {0: 0}
        match x:
            case {}:
                y = 0
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 0)

    def test_patma_024(self):
        x = {}
        y = None
        match x:
            case {0: 0}:
                y = 0
        self.assertEqual(x, {})
        self.assertIs(y, None)

    def test_patma_025(self):
        x = {0: 0}
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_026(self):
        x = {0: 1}
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 0)
        self.assertEqual(z, 1)

    def test_patma_027(self):
        x = {0: 2}
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 2})
        self.assertEqual(y, 0)
        self.assertEqual(z, 2)

    def test_patma_028(self):
        x = {0: 3}
        y = None
        match x:
            case {0: (0 | 1 | 2 as z)}:
                y = 0
        self.assertEqual(x, {0: 3})
        self.assertIs(y, None)

    def test_patma_029(self):
        x = {}
        y = None
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {})
        self.assertIs(y, None)

    def test_patma_030(self):
        x = {False: (True, 2.0, {})}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {False: (True, 2.0, {})})
        self.assertEqual(y, 0)

    def test_patma_031(self):
        x = {False: (True, 2.0, {}), 1: [[]], 2: 0}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {False: (True, 2.0, {}), 1: [[]], 2: 0})
        self.assertEqual(y, 0)

    def test_patma_032(self):
        x = {False: (True, 2.0, {}), 1: [[]], 2: 0}
        match x:
            case {0: [1, 2]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {False: (True, 2.0, {}), 1: [[]], 2: 0})
        self.assertEqual(y, 1)

    def test_patma_033(self):
        x = []
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}], 1: [[]]}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, [])
        self.assertEqual(y, 2)

    def test_patma_034(self):
        x = {0: 0}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: ([1, 2, {}] | False)} | {1: [[]]} | {0: [1, 2, {}]} | [] | "X" | {}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 1)

    def test_patma_035(self):
        x = {0: 0}
        match x:
            case {0: [1, 2, {}]}:
                y = 0
            case {0: [1, 2, {}] | True} | {1: [[]]} | {0: [1, 2, {}]} | [] | "X" | {}:
                y = 1
            case []:
                y = 2
        self.assertEqual(x, {0: 0})
        self.assertEqual(y, 1)

    def test_patma_036(self):
        x = 0
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_037(self):
        x = 1
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)

    def test_patma_038(self):
        x = 2
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

    def test_patma_039(self):
        x = 3
        y = None
        match x:
            case 0 | 1 | 2:
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_040(self):
        x = 0
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_041(self):
        x = 1
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)
        self.assertEqual(z, 1)

    def test_patma_042(self):
        x = 2
        y = None
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 2)
        self.assertIs(y, None)
        self.assertEqual(z, 2)

    def test_patma_043(self):
        x = 3
        y = None
        match x:
            case (0 as z) | (1 as z) | (2 as z) if z == x % 2:
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_044(self):
        x = ()
        match x:
            case []:
                y = 0
        self.assertEqual(x, ())
        self.assertEqual(y, 0)

    def test_patma_045(self):
        x = ()
        match x:
            case ():
                y = 0
        self.assertEqual(x, ())
        self.assertEqual(y, 0)

    def test_patma_046(self):
        x = (0,)
        match x:
            case [0]:
                y = 0
        self.assertEqual(x, (0,))
        self.assertEqual(y, 0)

    def test_patma_047(self):
        x = ((),)
        match x:
            case [[]]:
                y = 0
        self.assertEqual(x, ((),))
        self.assertEqual(y, 0)

    def test_patma_048(self):
        x = [0, 1]
        match x:
            case [0, 1] | [1, 0]:
                y = 0
        self.assertEqual(x, [0, 1])
        self.assertEqual(y, 0)

    def test_patma_049(self):
        x = [1, 0]
        match x:
            case [0, 1] | [1, 0]:
                y = 0
        self.assertEqual(x, [1, 0])
        self.assertEqual(y, 0)

    def test_patma_050(self):
        x = [0, 0]
        y = None
        match x:
            case [0, 1] | [1, 0]:
                y = 0
        self.assertEqual(x, [0, 0])
        self.assertIs(y, None)

    def test_patma_051(self):
        w = None
        x = [1, 0]
        match x:
            case [(0 as w)]:
                y = 0
            case [z] | [1, (0 | 1 as z)] | [z]:
                y = 1
        self.assertIs(w, None)
        self.assertEqual(x, [1, 0])
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_052(self):
        x = [1, 0]
        match x:
            case [0]:
                y = 0
            case [1, 0] if (x := x[:0]):
                y = 1
            case [1, 0]:
                y = 2
        self.assertEqual(x, [])
        self.assertEqual(y, 2)

    def test_patma_053(self):
        x = {0}
        y = None
        match x:
            case [0]:
                y = 0
        self.assertEqual(x, {0})
        self.assertIs(y, None)

    def test_patma_054(self):
        x = set()
        y = None
        match x:
            case []:
                y = 0
        self.assertEqual(x, set())
        self.assertIs(y, None)

    def test_patma_055(self):
        x = iter([1, 2, 3])
        y = None
        match x:
            case []:
                y = 0
        self.assertEqual([*x], [1, 2, 3])
        self.assertIs(y, None)

    def test_patma_056(self):
        x = {}
        y = None
        match x:
            case []:
                y = 0
        self.assertEqual(x, {})
        self.assertIs(y, None)

    def test_patma_057(self):
        x = {0: False, 1: True}
        y = None
        match x:
            case [0, 1]:
                y = 0
        self.assertEqual(x, {0: False, 1: True})
        self.assertIs(y, None)

    def test_patma_058(self):
        x = 0
        match x:
            case 0:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_059(self):
        x = 0
        y = None
        match x:
            case False:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, None)

    def test_patma_060(self):
        x = 0
        y = None
        match x:
            case 1:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_061(self):
        x = 0
        y = None
        match x:
            case None:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_062(self):
        x = 0
        match x:
            case 0:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_063(self):
        x = 0
        y = None
        match x:
            case 1:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_064(self):
        x = "x"
        match x:
            case "x":
                y = 0
            case "y":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 0)

    def test_patma_065(self):
        x = "x"
        match x:
            case "y":
                y = 0
            case "x":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 1)

    def test_patma_066(self):
        x = "x"
        match x:
            case "":
                y = 0
            case "x":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 1)

    def test_patma_067(self):
        x = b"x"
        match x:
            case b"y":
                y = 0
            case b"x":
                y = 1
        self.assertEqual(x, b"x")
        self.assertEqual(y, 1)

    def test_patma_068(self):
        x = 0
        match x:
            case 0 if False:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_069(self):
        x = 0
        y = None
        match x:
            case 0 if 0:
                y = 0
            case 0 if 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_070(self):
        x = 0
        match x:
            case 0 if True:
                y = 0
            case 0 if True:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_071(self):
        x = 0
        match x:
            case 0 if 1:
                y = 0
            case 0 if 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_072(self):
        x = 0
        match x:
            case 0 if True:
                y = 0
            case 0 if True:
                y = 1
        y = 2
        self.assertEqual(x, 0)
        self.assertEqual(y, 2)

    def test_patma_073(self):
        x = 0
        match x:
            case 0 if 0:
                y = 0
            case 0 if 1:
                y = 1
        y = 2
        self.assertEqual(x, 0)
        self.assertEqual(y, 2)

    def test_patma_074(self):
        x = 0
        y = None
        match x:
            case 0 if not (x := 1):
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 1)
        self.assertIs(y, None)

    def test_patma_075(self):
        x = "x"
        match x:
            case ["x"]:
                y = 0
            case "x":
                y = 1
        self.assertEqual(x, "x")
        self.assertEqual(y, 1)

    def test_patma_076(self):
        x = b"x"
        match x:
            case [b"x"]:
                y = 0
            case ["x"]:
                y = 1
            case [120]:
                y = 2
            case b"x":
                y = 4
        self.assertEqual(x, b"x")
        self.assertEqual(y, 4)

    def test_patma_077(self):
        x = bytearray(b"x")
        y = None
        match x:
            case [120]:
                y = 0
            case 120:
                y = 1
        self.assertEqual(x, b"x")
        self.assertIs(y, None)

    def test_patma_078(self):
        x = ""
        match x:
            case []:
                y = 0
            case [""]:
                y = 1
            case "":
                y = 2
        self.assertEqual(x, "")
        self.assertEqual(y, 2)

    def test_patma_079(self):
        x = "xxx"
        match x:
            case ["x", "x", "x"]:
                y = 0
            case ["xxx"]:
                y = 1
            case "xxx":
                y = 2
        self.assertEqual(x, "xxx")
        self.assertEqual(y, 2)

    def test_patma_080(self):
        x = b"xxx"
        match x:
            case [120, 120, 120]:
                y = 0
            case [b"xxx"]:
                y = 1
            case b"xxx":
                y = 2
        self.assertEqual(x, b"xxx")
        self.assertEqual(y, 2)

    def test_patma_081(self):
        x = 0
        match x:
            case 0 if not (x := 1):
                y = 0
            case (0 as z):
                y = 1
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_082(self):
        x = 0
        match x:
            case (1 as z) if not (x := 1):
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_083(self):
        x = 0
        match x:
            case (0 as z):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_084(self):
        x = 0
        y = None
        match x:
            case (1 as z):
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_085(self):
        x = 0
        y = None
        match x:
            case (0 as z) if (w := 0):
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertIs(y, None)
        self.assertEqual(z, 0)

    def test_patma_086(self):
        x = 0
        match x:
            case ((0 as w) as z):
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_087(self):
        x = 0
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_088(self):
        x = 1
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)

    def test_patma_089(self):
        x = 2
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

    def test_patma_090(self):
        x = 3
        y = None
        match x:
            case (0 | 1) | 2:
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_091(self):
        x = 0
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_092(self):
        x = 1
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 1)
        self.assertEqual(y, 0)

    def test_patma_093(self):
        x = 2
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 2)
        self.assertEqual(y, 0)

    def test_patma_094(self):
        x = 3
        y = None
        match x:
            case 0 | (1 | 2):
                y = 0
        self.assertEqual(x, 3)
        self.assertIs(y, None)

    def test_patma_095(self):
        x = 0
        match x:
            case -0:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_096(self):
        x = 0
        match x:
            case -0.0:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_097(self):
        x = 0
        match x:
            case -0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_098(self):
        x = 0
        match x:
            case -0.0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_099(self):
        x = -1
        match x:
            case -1:
                y = 0
        self.assertEqual(x, -1)
        self.assertEqual(y, 0)

    def test_patma_100(self):
        x = -1.5
        match x:
            case -1.5:
                y = 0
        self.assertEqual(x, -1.5)
        self.assertEqual(y, 0)

    def test_patma_101(self):
        x = -1j
        match x:
            case -1j:
                y = 0
        self.assertEqual(x, -1j)
        self.assertEqual(y, 0)

    def test_patma_102(self):
        x = -1.5j
        match x:
            case -1.5j:
                y = 0
        self.assertEqual(x, -1.5j)
        self.assertEqual(y, 0)

    def test_patma_103(self):
        x = 0
        match x:
            case 0 + 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_104(self):
        x = 0
        match x:
            case 0 - 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_105(self):
        x = 0
        match x:
            case -0 + 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_106(self):
        x = 0
        match x:
            case -0 - 0j:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_107(self):
        x = 0.25 + 1.75j
        match x:
            case 0.25 + 1.75j:
                y = 0
        self.assertEqual(x, 0.25 + 1.75j)
        self.assertEqual(y, 0)

    def test_patma_108(self):
        x = 0.25 - 1.75j
        match x:
            case 0.25 - 1.75j:
                y = 0
        self.assertEqual(x, 0.25 - 1.75j)
        self.assertEqual(y, 0)

    def test_patma_109(self):
        x = -0.25 + 1.75j
        match x:
            case -0.25 + 1.75j:
                y = 0
        self.assertEqual(x, -0.25 + 1.75j)
        self.assertEqual(y, 0)

    def test_patma_110(self):
        x = -0.25 - 1.75j
        match x:
            case -0.25 - 1.75j:
                y = 0
        self.assertEqual(x, -0.25 - 1.75j)
        self.assertEqual(y, 0)

    def test_patma_111(self):
        class A:
            B = 0
        x = 0
        match x:
            case A.B:
                y = 0
        self.assertEqual(A.B, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_112(self):
        class A:
            class B:
                C = 0
        x = 0
        match x:
            case A.B.C:
                y = 0
        self.assertEqual(A.B.C, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_113(self):
        class A:
            class B:
                C = 0
                D = 1
        x = 1
        match x:
            case A.B.C:
                y = 0
            case A.B.D:
                y = 1
        self.assertEqual(A.B.C, 0)
        self.assertEqual(A.B.D, 1)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_patma_114(self):
        class A:
            class B:
                class C:
                    D = 0
        x = 0
        match x:
            case A.B.C.D:
                y = 0
        self.assertEqual(A.B.C.D, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_115(self):
        class A:
            class B:
                class C:
                    D = 0
                    E = 1
        x = 1
        match x:
            case A.B.C.D:
                y = 0
            case A.B.C.E:
                y = 1
        self.assertEqual(A.B.C.D, 0)
        self.assertEqual(A.B.C.E, 1)
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_patma_116(self):
        match = case = 0
        match match:
            case case:
                x = 0
        self.assertEqual(match, 0)
        self.assertEqual(case, 0)
        self.assertEqual(x, 0)

    def test_patma_117(self):
        match = case = 0
        match case:
            case match:
                x = 0
        self.assertEqual(match, 0)
        self.assertEqual(case, 0)
        self.assertEqual(x, 0)

    def test_patma_118(self):
        x = []
        match x:
            case [*_, _]:
                y = 0
            case []:
                y = 1
        self.assertEqual(x, [])
        self.assertEqual(y, 1)

    def test_patma_119(self):
        x = collections.defaultdict(int)
        match x:
            case {0: 0}:
                y = 0
            case {}:
                y = 1
        self.assertEqual(x, {})
        self.assertEqual(y, 1)

    def test_patma_120(self):
        x = collections.defaultdict(int)
        match x:
            case {0: 0}:
                y = 0
            case {**z}:
                y = 1
        self.assertEqual(x, {})
        self.assertEqual(y, 1)
        self.assertEqual(z, {})

    def test_patma_121(self):
        match ():
            case ():
                x = 0
        self.assertEqual(x, 0)

    def test_patma_122(self):
        match (0, 1, 2):
            case (*x,):
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_123(self):
        match (0, 1, 2):
            case 0, *x:
                y = 0
        self.assertEqual(x, [1, 2])
        self.assertEqual(y, 0)

    def test_patma_124(self):
        match (0, 1, 2):
            case (0, 1, *x,):
                y = 0
        self.assertEqual(x, [2])
        self.assertEqual(y, 0)

    def test_patma_125(self):
        match (0, 1, 2):
            case 0, 1, 2, *x:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_126(self):
        match (0, 1, 2):
            case *x, 2,:
                y = 0
        self.assertEqual(x, [0, 1])
        self.assertEqual(y, 0)

    def test_patma_127(self):
        match (0, 1, 2):
            case (*x, 1, 2):
                y = 0
        self.assertEqual(x, [0])
        self.assertEqual(y, 0)

    def test_patma_128(self):
        match (0, 1, 2):
            case *x, 0, 1, 2,:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_129(self):
        match (0, 1, 2):
            case (0, *x, 2):
                y = 0
        self.assertEqual(x, [1])
        self.assertEqual(y, 0)

    def test_patma_130(self):
        match (0, 1, 2):
            case 0, 1, *x, 2,:
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_131(self):
        match (0, 1, 2):
            case (0, *x, 1, 2):
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)

    def test_patma_132(self):
        match (0, 1, 2):
            case *x,:
                y = 0
        self.assertEqual(x, [0, 1, 2])
        self.assertEqual(y, 0)

    def test_patma_133(self):
        x = collections.defaultdict(int, {0: 1})
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 1
            case {}:
                y = 2
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 2)

    def test_patma_134(self):
        x = collections.defaultdict(int, {0: 1})
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 1
            case {**z}:
                y = 2
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 2)
        self.assertEqual(z, {0: 1})

    def test_patma_135(self):
        x = collections.defaultdict(int, {0: 1})
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 1
            case {0: _, **z}:
                y = 2
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 2)
        self.assertEqual(z, {})

    def test_patma_136(self):
        x = {0: 1}
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 0
            case {}:
                y = 1
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 1)

    def test_patma_137(self):
        x = {0: 1}
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 0
            case {**z}:
                y = 1
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 1)
        self.assertEqual(z, {0: 1})

    def test_patma_138(self):
        x = {0: 1}
        match x:
            case {1: 0}:
                y = 0
            case {0: 0}:
                y = 0
            case {0: _, **z}:
                y = 1
        self.assertEqual(x, {0: 1})
        self.assertEqual(y, 1)
        self.assertEqual(z, {})

    def test_patma_139(self):
        x = False
        match x:
            case bool(z):
                y = 0
        self.assertIs(x, False)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_140(self):
        x = True
        match x:
            case bool(z):
                y = 0
        self.assertIs(x, True)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_141(self):
        x = bytearray()
        match x:
            case bytearray(z):
                y = 0
        self.assertEqual(x, bytearray())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_142(self):
        x = b""
        match x:
            case bytes(z):
                y = 0
        self.assertEqual(x, b"")
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_143(self):
        x = {}
        match x:
            case dict(z):
                y = 0
        self.assertEqual(x, {})
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_144(self):
        x = 0.0
        match x:
            case float(z):
                y = 0
        self.assertEqual(x, 0.0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_145(self):
        x = frozenset()
        match x:
            case frozenset(z):
                y = 0
        self.assertEqual(x, frozenset())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_146(self):
        x = 0
        match x:
            case int(z):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_147(self):
        x = []
        match x:
            case list(z):
                y = 0
        self.assertEqual(x, [])
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_148(self):
        x = set()
        match x:
            case set(z):
                y = 0
        self.assertEqual(x, set())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_149(self):
        x = ""
        match x:
            case str(z):
                y = 0
        self.assertEqual(x, "")
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_150(self):
        x = ()
        match x:
            case tuple(z):
                y = 0
        self.assertEqual(x, ())
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_151(self):
        x = 0
        match x,:
            case y,:
                z = 0
        self.assertEqual(x, 0)
        self.assertIs(y, x)
        self.assertIs(z, 0)

    def test_patma_152(self):
        w = 0
        x = 0
        match w, x:
            case y, z:
                v = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertIs(y, w)
        self.assertIs(z, x)
        self.assertEqual(v, 0)

    def test_patma_153(self):
        x = 0
        match w := x,:
            case y as v,:
                z = 0
        self.assertEqual(x, 0)
        self.assertIs(y, x)
        self.assertEqual(z, 0)
        self.assertIs(w, x)
        self.assertIs(v, y)

    def test_patma_154(self):
        x = 0
        y = None
        match x:
            case 0 if x:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_155(self):
        x = 0
        y = None
        match x:
            case 1e1000:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_156(self):
        x = 0
        match x:
            case z:
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_157(self):
        x = 0
        y = None
        match x:
            case _ if x:
                y = 0
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_158(self):
        x = 0
        match x:
            case -1e1000:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_159(self):
        x = 0
        match x:
            case 0 if not x:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_160(self):
        x = 0
        z = None
        match x:
            case 0:
                y = 0
            case z if x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, None)

    def test_patma_161(self):
        x = 0
        match x:
            case 0:
                y = 0
            case _:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_162(self):
        x = 0
        match x:
            case 1 if x:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_163(self):
        x = 0
        y = None
        match x:
            case 1:
                y = 0
            case 1 if not x:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_164(self):
        x = 0
        match x:
            case 1:
                y = 0
            case z:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertIs(z, x)

    def test_patma_165(self):
        x = 0
        match x:
            case 1 if x:
                y = 0
            case _:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_166(self):
        x = 0
        match x:
            case z if not z:
                y = 0
            case 0 if x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_167(self):
        x = 0
        match x:
            case z if not z:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_168(self):
        x = 0
        match x:
            case z if not x:
                y = 0
            case z:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_169(self):
        x = 0
        match x:
            case z if not z:
                y = 0
            case _ if x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, x)

    def test_patma_170(self):
        x = 0
        match x:
            case _ if not x:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_171(self):
        x = 0
        y = None
        match x:
            case _ if x:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 0)
        self.assertIs(y, None)

    def test_patma_172(self):
        x = 0
        z = None
        match x:
            case _ if not x:
                y = 0
            case z if not x:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertIs(z, None)

    def test_patma_173(self):
        x = 0
        match x:
            case _ if not x:
                y = 0
            case _:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_174(self):
        def http_error(status):
            match status:
                case 400:
                    return "Bad request"
                case 401:
                    return "Unauthorized"
                case 403:
                    return "Forbidden"
                case 404:
                    return "Not found"
                case 418:
                    return "I'm a teapot"
                case _:
                    return "Something else"
        self.assertEqual(http_error(400), "Bad request")
        self.assertEqual(http_error(401), "Unauthorized")
        self.assertEqual(http_error(403), "Forbidden")
        self.assertEqual(http_error(404), "Not found")
        self.assertEqual(http_error(418), "I'm a teapot")
        self.assertEqual(http_error(123), "Something else")
        self.assertEqual(http_error("400"), "Something else")
        self.assertEqual(http_error(401 | 403 | 404), "Something else")  # 407

    def test_patma_175(self):
        def http_error(status):
            match status:
                case 400:
                    return "Bad request"
                case 401 | 403 | 404:
                    return "Not allowed"
                case 418:
                    return "I'm a teapot"
        self.assertEqual(http_error(400), "Bad request")
        self.assertEqual(http_error(401), "Not allowed")
        self.assertEqual(http_error(403), "Not allowed")
        self.assertEqual(http_error(404), "Not allowed")
        self.assertEqual(http_error(418), "I'm a teapot")
        self.assertIs(http_error(123), None)
        self.assertIs(http_error("400"), None)
        self.assertIs(http_error(401 | 403 | 404), None)  # 407

    def test_patma_176(self):
        def whereis(point):
            match point:
                case (0, 0):
                    return "Origin"
                case (0, y):
                    return f"Y={y}"
                case (x, 0):
                    return f"X={x}"
                case (x, y):
                    return f"X={x}, Y={y}"
                case _:
                    return "Not a point"
        self.assertEqual(whereis((0, 0)), "Origin")
        self.assertEqual(whereis((0, -1.0)), "Y=-1.0")
        self.assertEqual(whereis(("X", 0)), "X=X")
        self.assertEqual(whereis((None, 1j)), "X=None, Y=1j")
        self.assertEqual(whereis(42), "Not a point")

    def test_patma_177(self):
        def whereis(point):
            match point:
                case Point(0, 0):
                    return "Origin"
                case Point(0, y):
                    return f"Y={y}"
                case Point(x, 0):
                    return f"X={x}"
                case Point():
                    return "Somewhere else"
                case _:
                    return "Not a point"
        self.assertEqual(whereis(Point(1, 0)), "X=1")
        self.assertEqual(whereis(Point(0, 0)), "Origin")
        self.assertEqual(whereis(10), "Not a point")
        self.assertEqual(whereis(Point(False, False)), "Origin")
        self.assertEqual(whereis(Point(0, -1.0)), "Y=-1.0")
        self.assertEqual(whereis(Point("X", 0)), "X=X")
        self.assertEqual(whereis(Point(None, 1j)), "Somewhere else")
        self.assertEqual(whereis(Point), "Not a point")
        self.assertEqual(whereis(42), "Not a point")

    def test_patma_178(self):
        def whereis(point):
            match point:
                case Point(1, var):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_179(self):
        def whereis(point):
            match point:
                case Point(1, y=var):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_180(self):
        def whereis(point):
            match point:
                case Point(x=1, y=var):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_181(self):
        def whereis(point):
            match point:
                case Point(y=var, x=1):
                    return var
        self.assertEqual(whereis(Point(1, 0)), 0)
        self.assertIs(whereis(Point(0, 0)), None)

    def test_patma_182(self):
        def whereis(points):
            match points:
                case []:
                    return "No points"
                case [Point(0, 0)]:
                    return "The origin"
                case [Point(x, y)]:
                    return f"Single point {x}, {y}"
                case [Point(0, y1), Point(0, y2)]:
                    return f"Two on the Y axis at {y1}, {y2}"
                case _:
                    return "Something else"
        self.assertEqual(whereis([]), "No points")
        self.assertEqual(whereis([Point(0, 0)]), "The origin")
        self.assertEqual(whereis([Point(0, 1)]), "Single point 0, 1")
        self.assertEqual(whereis([Point(0, 0), Point(0, 0)]), "Two on the Y axis at 0, 0")
        self.assertEqual(whereis([Point(0, 1), Point(0, 1)]), "Two on the Y axis at 1, 1")
        self.assertEqual(whereis([Point(0, 0), Point(1, 0)]), "Something else")
        self.assertEqual(whereis([Point(0, 0), Point(0, 0), Point(0, 0)]), "Something else")
        self.assertEqual(whereis([Point(0, 1), Point(0, 1), Point(0, 1)]), "Something else")

    def test_patma_183(self):
        def whereis(point):
            match point:
                case Point(x, y) if x == y:
                    return f"Y=X at {x}"
                case Point(x, y):
                    return "Not on the diagonal"
        self.assertEqual(whereis(Point(0, 0)), "Y=X at 0")
        self.assertEqual(whereis(Point(0, False)), "Y=X at 0")
        self.assertEqual(whereis(Point(False, 0)), "Y=X at False")
        self.assertEqual(whereis(Point(-1 - 1j, -1 - 1j)), "Y=X at (-1-1j)")
        self.assertEqual(whereis(Point("X", "X")), "Y=X at X")
        self.assertEqual(whereis(Point("X", "x")), "Not on the diagonal")

    def test_patma_184(self):
        class Seq(collections.abc.Sequence):
            __getitem__ = None
            def __len__(self):
                return 0
        match Seq():
            case []:
                y = 0
        self.assertEqual(y, 0)

    def test_patma_185(self):
        class Seq(collections.abc.Sequence):
            __getitem__ = None
            def __len__(self):
                return 42
        match Seq():
            case [*_]:
                y = 0
        self.assertEqual(y, 0)

    def test_patma_186(self):
        class Seq(collections.abc.Sequence):
            def __getitem__(self, i):
                return i
            def __len__(self):
                return 42
        match Seq():
            case [x, *_, y]:
                z = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 41)
        self.assertEqual(z, 0)

    def test_patma_187(self):
        w = range(10)
        match w:
            case [x, y, *rest]:
                z = 0
        self.assertEqual(w, range(10))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)
        self.assertEqual(rest, list(range(2, 10)))

    def test_patma_188(self):
        w = range(100)
        match w:
            case (x, y, *rest):
                z = 0
        self.assertEqual(w, range(100))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)
        self.assertEqual(rest, list(range(2, 100)))

    def test_patma_189(self):
        w = range(1000)
        match w:
            case x, y, *rest:
                z = 0
        self.assertEqual(w, range(1000))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)
        self.assertEqual(rest, list(range(2, 1000)))

    def test_patma_190(self):
        w = range(1 << 10)
        match w:
            case [x, y, *_]:
                z = 0
        self.assertEqual(w, range(1 << 10))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_191(self):
        w = range(1 << 20)
        match w:
            case (x, y, *_):
                z = 0
        self.assertEqual(w, range(1 << 20))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_192(self):
        w = range(1 << 30)
        match w:
            case x, y, *_:
                z = 0
        self.assertEqual(w, range(1 << 30))
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_193(self):
        x = {"bandwidth": 0, "latency": 1}
        match x:
            case {"bandwidth": b, "latency": l}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1})
        self.assertIs(b, x["bandwidth"])
        self.assertIs(l, x["latency"])
        self.assertEqual(y, 0)

    def test_patma_194(self):
        x = {"bandwidth": 0, "latency": 1, "key": "value"}
        match x:
            case {"latency": l, "bandwidth": b}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1, "key": "value"})
        self.assertIs(l, x["latency"])
        self.assertIs(b, x["bandwidth"])
        self.assertEqual(y, 0)

    def test_patma_195(self):
        x = {"bandwidth": 0, "latency": 1, "key": "value"}
        match x:
            case {"bandwidth": b, "latency": l, **rest}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1, "key": "value"})
        self.assertIs(b, x["bandwidth"])
        self.assertIs(l, x["latency"])
        self.assertEqual(rest, {"key": "value"})
        self.assertEqual(y, 0)

    def test_patma_196(self):
        x = {"bandwidth": 0, "latency": 1}
        match x:
            case {"latency": l, "bandwidth": b, **rest}:
                y = 0
        self.assertEqual(x, {"bandwidth": 0, "latency": 1})
        self.assertIs(l, x["latency"])
        self.assertIs(b, x["bandwidth"])
        self.assertEqual(rest, {})
        self.assertEqual(y, 0)

    def test_patma_197(self):
        w = [Point(-1, 0), Point(1, 2)]
        match w:
            case (Point(x1, y1), Point(x2, y2) as p2):
                z = 0
        self.assertEqual(w, [Point(-1, 0), Point(1, 2)])
        self.assertIs(x1, w[0].x)
        self.assertIs(y1, w[0].y)
        self.assertIs(p2, w[1])
        self.assertIs(x2, w[1].x)
        self.assertIs(y2, w[1].y)
        self.assertIs(z, 0)

    def test_patma_198(self):
        class Color(enum.Enum):
            RED = 0
            GREEN = 1
            BLUE = 2
        def f(color):
            match color:
                case Color.RED:
                    return "I see red!"
                case Color.GREEN:
                    return "Grass is green"
                case Color.BLUE:
                    return "I'm feeling the blues :("
        self.assertEqual(f(Color.RED), "I see red!")
        self.assertEqual(f(Color.GREEN), "Grass is green")
        self.assertEqual(f(Color.BLUE), "I'm feeling the blues :(")
        self.assertIs(f(Color), None)
        self.assertIs(f(0), None)
        self.assertIs(f(1), None)
        self.assertIs(f(2), None)
        self.assertIs(f(3), None)
        self.assertIs(f(False), None)
        self.assertIs(f(True), None)
        self.assertIs(f(2+0j), None)
        self.assertIs(f(3.0), None)

    def test_patma_199(self):
        class Color(int, enum.Enum):
            RED = 0
            GREEN = 1
            BLUE = 2
        def f(color):
            match color:
                case Color.RED:
                    return "I see red!"
                case Color.GREEN:
                    return "Grass is green"
                case Color.BLUE:
                    return "I'm feeling the blues :("
        self.assertEqual(f(Color.RED), "I see red!")
        self.assertEqual(f(Color.GREEN), "Grass is green")
        self.assertEqual(f(Color.BLUE), "I'm feeling the blues :(")
        self.assertIs(f(Color), None)
        self.assertEqual(f(0), "I see red!")
        self.assertEqual(f(1), "Grass is green")
        self.assertEqual(f(2), "I'm feeling the blues :(")
        self.assertIs(f(3), None)
        self.assertEqual(f(False), "I see red!")
        self.assertEqual(f(True), "Grass is green")
        self.assertEqual(f(2+0j), "I'm feeling the blues :(")
        self.assertIs(f(3.0), None)

    def test_patma_200(self):
        class Class:
            __match_args__ = ("a", "b")
        c = Class()
        c.a = 0
        c.b = 1
        match c:
            case Class(x, y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_201(self):
        class Class:
            __match_args__ = ("a", "b")
        c = Class()
        c.a = 0
        c.b = 1
        match c:
            case Class(x, b=y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_202(self):
        class Parent:
            __match_args__ = "a", "b"
        class Child(Parent):
            __match_args__ = ("c", "d")
        c = Child()
        c.a = 0
        c.b = 1
        match c:
            case Parent(x, y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_203(self):
        class Parent:
            __match_args__ = ("a", "b")
        class Child(Parent):
            __match_args__ = "c", "d"
        c = Child()
        c.a = 0
        c.b = 1
        match c:
            case Parent(x, b=y):
                z = 0
        self.assertIs(x, c.a)
        self.assertIs(y, c.b)
        self.assertEqual(z, 0)

    def test_patma_204(self):
        def f(w):
            match w:
                case 42:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42), {})
        self.assertIs(f(0), None)
        self.assertEqual(f(42.0), {})
        self.assertIs(f("42"), None)

    def test_patma_205(self):
        def f(w):
            match w:
                case 42.0:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42.0), {})
        self.assertEqual(f(42), {})
        self.assertIs(f(0.0), None)
        self.assertIs(f(0), None)

    def test_patma_206(self):
        def f(w):
            match w:
                case 1 | 2 | 3:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(1), {})
        self.assertEqual(f(2), {})
        self.assertEqual(f(3), {})
        self.assertEqual(f(3.0), {})
        self.assertIs(f(0), None)
        self.assertIs(f(4), None)
        self.assertIs(f("1"), None)

    def test_patma_207(self):
        def f(w):
            match w:
                case [1, 2] | [3, 4]:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f([1, 2]), {})
        self.assertEqual(f([3, 4]), {})
        self.assertIs(f(42), None)
        self.assertIs(f([2, 3]), None)
        self.assertIs(f([1, 2, 3]), None)
        self.assertEqual(f([1, 2.0]), {})

    def test_patma_208(self):
        def f(w):
            match w:
                case x:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42), {"x": 42})
        self.assertEqual(f((1, 2)), {"x": (1, 2)})
        self.assertEqual(f(None), {"x": None})

    def test_patma_209(self):
        def f(w):
            match w:
                case _:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(42), {})
        self.assertEqual(f(None), {})
        self.assertEqual(f((1, 2)), {})

    def test_patma_210(self):
        def f(w):
            match w:
                case (x, y, z):
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f((1, 2, 3)), {"x": 1, "y": 2, "z": 3})
        self.assertIs(f((1, 2)), None)
        self.assertIs(f((1, 2, 3, 4)), None)
        self.assertIs(f(123), None)
        self.assertIs(f("abc"), None)
        self.assertIs(f(b"abc"), None)
        self.assertEqual(f(array.array("b", b"abc")), {'x': 97, 'y': 98, 'z': 99})
        self.assertEqual(f(memoryview(b"abc")), {"x": 97, "y": 98, "z": 99})
        self.assertIs(f(bytearray(b"abc")), None)

    def test_patma_211(self):
        def f(w):
            match w:
                case {"x": x, "y": "y", "z": z}:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f({"x": "x", "y": "y", "z": "z"}), {"x": "x", "z": "z"})
        self.assertEqual(f({"x": "x", "y": "y", "z": "z", "a": "a"}), {"x": "x", "z": "z"})
        self.assertIs(f(({"x": "x", "y": "yy", "z": "z", "a": "a"})), None)
        self.assertIs(f(({"x": "x", "y": "y"})), None)

    def test_patma_212(self):
        def f(w):
            match w:
                case Point(int(xx), y="hello"):
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f(Point(42, "hello")), {"xx": 42})

    def test_patma_213(self):
        def f(w):
            match w:
                case (p, q) as x:
                    out = locals()
                    del out["w"]
                    return out
        self.assertEqual(f((1, 2)), {"p": 1, "q": 2, "x": (1, 2)})
        self.assertEqual(f([1, 2]), {"p": 1, "q": 2, "x": [1, 2]})
        self.assertIs(f(12), None)
        self.assertIs(f((1, 2, 3)), None)

    def test_patma_214(self):
        def f():
            match 42:
                case 42:
                    return locals()
        self.assertEqual(set(f()), set())

    def test_patma_215(self):
        def f():
            match 1:
                case 1 | 2 | 3:
                    return locals()
        self.assertEqual(set(f()), set())

    def test_patma_216(self):
        def f():
            match ...:
                case _:
                    return locals()
        self.assertEqual(set(f()), set())

    def test_patma_217(self):
        def f():
            match ...:
                case abc:
                    return locals()
        self.assertEqual(set(f()), {"abc"})

    def test_patma_218(self):
        def f():
            match ..., ...:
                case a, b:
                    return locals()
        self.assertEqual(set(f()), {"a", "b"})

    def test_patma_219(self):
        def f():
            match {"k": ..., "l": ...}:
                case {"k": a, "l": b}:
                    return locals()
        self.assertEqual(set(f()), {"a", "b"})

    def test_patma_220(self):
        def f():
            match Point(..., ...):
                case Point(x, y=y):
                    return locals()
        self.assertEqual(set(f()), {"x", "y"})

    def test_patma_221(self):
        def f():
            match ...:
                case b as a:
                    return locals()
        self.assertEqual(set(f()), {"a", "b"})

    def test_patma_222(self):
        def f(x):
            match x:
                case _:
                    return 0
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 0)
        self.assertEqual(f(2), 0)
        self.assertEqual(f(3), 0)

    def test_patma_223(self):
        def f(x):
            match x:
                case 0:
                    return 0
        self.assertEqual(f(0), 0)
        self.assertIs(f(1), None)
        self.assertIs(f(2), None)
        self.assertIs(f(3), None)

    def test_patma_224(self):
        def f(x):
            match x:
                case 0:
                    return 0
                case _:
                    return 1
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 1)
        self.assertEqual(f(3), 1)

    def test_patma_225(self):
        def f(x):
            match x:
                case 0:
                    return 0
                case 1:
                    return 1
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertIs(f(2), None)
        self.assertIs(f(3), None)

    def test_patma_226(self):
        def f(x):
            match x:
                case 0:
                    return 0
                case 1:
                    return 1
                case _:
                    return 2
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertEqual(f(3), 2)

    def test_patma_227(self):
        def f(x):
            match x:
                case 0:
                    return 0
                case 1:
                    return 1
                case 2:
                    return 2
        self.assertEqual(f(0), 0)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 2)
        self.assertIs(f(3), None)

    def test_patma_228(self):
        match():
            case():
                x = 0
        self.assertEqual(x, 0)

    def test_patma_229(self):
        x = 0
        match(x):
            case(x):
                y = 0
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_patma_230(self):
        x = 0
        match x:
            case False:
                y = 0
            case 0:
                y = 1
        self.assertEqual(x, 0)
        self.assertEqual(y, 1)

    def test_patma_231(self):
        x = 1
        match x:
            case True:
                y = 0
            case 1:
                y = 1
        self.assertEqual(x, 1)
        self.assertEqual(y, 1)

    def test_patma_232(self):
        class Eq:
            def __eq__(self, other):
                return True
        x = eq = Eq()
        # None
        y = None
        match x:
            case None:
                y = 0
        self.assertIs(x, eq)
        self.assertEqual(y, None)
        # True
        y = None
        match x:
            case True:
                y = 0
        self.assertIs(x, eq)
        self.assertEqual(y, None)
        # False
        y = None
        match x:
            case False:
                y = 0
        self.assertIs(x, eq)
        self.assertEqual(y, None)

    def test_patma_233(self):
        x = False
        match x:
            case False:
                y = 0
        self.assertIs(x, False)
        self.assertEqual(y, 0)

    def test_patma_234(self):
        x = True
        match x:
            case True:
                y = 0
        self.assertIs(x, True)
        self.assertEqual(y, 0)

    def test_patma_235(self):
        x = None
        match x:
            case None:
                y = 0
        self.assertIs(x, None)
        self.assertEqual(y, 0)

    def test_patma_236(self):
        x = 0
        match x:
            case (0 as w) as z:
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_237(self):
        x = 0
        match x:
            case (0 as w) as z:
                y = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_238(self):
        x = ((0, 1), (2, 3))
        match x:
            case ((a as b, c as d) as e) as w, ((f as g, h) as i) as z:
                y = 0
        self.assertEqual(a, 0)
        self.assertEqual(b, 0)
        self.assertEqual(c, 1)
        self.assertEqual(d, 1)
        self.assertEqual(e, (0, 1))
        self.assertEqual(f, 2)
        self.assertEqual(g, 2)
        self.assertEqual(h, 3)
        self.assertEqual(i, (2, 3))
        self.assertEqual(w, (0, 1))
        self.assertEqual(x, ((0, 1), (2, 3)))
        self.assertEqual(y, 0)
        self.assertEqual(z, (2, 3))

    def test_patma_239(self):
        x = collections.UserDict({0: 1, 2: 3})
        match x:
            case {2: 3}:
                y = 0
        self.assertEqual(x, {0: 1, 2: 3})
        self.assertEqual(y, 0)

    def test_patma_240(self):
        x = collections.UserDict({0: 1, 2: 3})
        match x:
            case {2: 3, **z}:
                y = 0
        self.assertEqual(x, {0: 1, 2: 3})
        self.assertEqual(y, 0)
        self.assertEqual(z, {0: 1})

    def test_patma_241(self):
        x = [[{0: 0}]]
        match x:
            case list([({-0-0j: int(real=0+0j, imag=0-0j) | (1) as z},)]):
                y = 0
        self.assertEqual(x, [[{0: 0}]])
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_patma_242(self):
        x = range(3)
        match x:
            case [y, *_, z]:
                w = 0
        self.assertEqual(w, 0)
        self.assertEqual(x, range(3))
        self.assertEqual(y, 0)
        self.assertEqual(z, 2)

    def test_patma_243(self):
        x = range(3)
        match x:
            case [_, *_, y]:
                z = 0
        self.assertEqual(x, range(3))
        self.assertEqual(y, 2)
        self.assertEqual(z, 0)

    def test_patma_244(self):
        x = range(3)
        match x:
            case [*_, y]:
                z = 0
        self.assertEqual(x, range(3))
        self.assertEqual(y, 2)
        self.assertEqual(z, 0)

    def test_patma_245(self):
        x = {"y": 1}
        match x:
            case {"y": (0 as y) | (1 as y)}:
                z = 0
        self.assertEqual(x, {"y": 1})
        self.assertEqual(y, 1)
        self.assertEqual(z, 0)

    def test_patma_246(self):
        def f(x):
            match x:
                case ((a, b, c, d, e, f, g, h, i, 9) |
                      (h, g, i, a, b, d, e, c, f, 10) |
                      (g, b, a, c, d, -5, e, h, i, f) |
                      (-1, d, f, b, g, e, i, a, h, c)):
                    w = 0
            out = locals()
            del out["x"]
            return out
        alts = [
            dict(a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, w=0),
            dict(h=1, g=2, i=3, a=4, b=5, d=6, e=7, c=8, f=9, w=0),
            dict(g=0, b=-1, a=-2, c=-3, d=-4, e=-6, h=-7, i=-8, f=-9, w=0),
            dict(d=-2, f=-3, b=-4, g=-5, e=-6, i=-7, a=-8, h=-9, c=-10, w=0),
            dict(),
        ]
        self.assertEqual(f(range(10)), alts[0])
        self.assertEqual(f(range(1, 11)), alts[1])
        self.assertEqual(f(range(0, -10, -1)), alts[2])
        self.assertEqual(f(range(-1, -11, -1)), alts[3])
        self.assertEqual(f(range(10, 20)), alts[4])

    def test_patma_247(self):
        def f(x):
            match x:
                case [y, (a, b, c, d, e, f, g, h, i, 9) |
                         (h, g, i, a, b, d, e, c, f, 10) |
                         (g, b, a, c, d, -5, e, h, i, f) |
                         (-1, d, f, b, g, e, i, a, h, c), z]:
                    w = 0
            out = locals()
            del out["x"]
            return out
        alts = [
            dict(a=0, b=1, c=2, d=3, e=4, f=5, g=6, h=7, i=8, w=0, y=False, z=True),
            dict(h=1, g=2, i=3, a=4, b=5, d=6, e=7, c=8, f=9, w=0, y=False, z=True),
            dict(g=0, b=-1, a=-2, c=-3, d=-4, e=-6, h=-7, i=-8, f=-9, w=0, y=False, z=True),
            dict(d=-2, f=-3, b=-4, g=-5, e=-6, i=-7, a=-8, h=-9, c=-10, w=0, y=False, z=True),
            dict(),
        ]
        self.assertEqual(f((False, range(10), True)), alts[0])
        self.assertEqual(f((False, range(1, 11), True)), alts[1])
        self.assertEqual(f((False, range(0, -10, -1), True)), alts[2])
        self.assertEqual(f((False, range(-1, -11, -1), True)), alts[3])
        self.assertEqual(f((False, range(10, 20), True)), alts[4])

    def test_patma_248(self):
        class C(dict):
            @staticmethod
            def get(key, default=None):
                return 'bar'

        x = C({'foo': 'bar'})
        match x:
            case {'foo': bar}:
                y = bar

        self.assertEqual(y, 'bar')

    def test_patma_249(self):
        class C:
            __attr = "eggs"  # mangled to _C__attr
            _Outer__attr = "bacon"
        class Outer:
            def f(self, x):
                match x:
                    # looks up __attr, not _C__attr or _Outer__attr
                    case C(__attr=y):
                        return y
        c = C()
        setattr(c, "__attr", "spam")  # setattr is needed because we're in a class scope
        self.assertEqual(Outer().f(c), "spam")

    def test_patma_250(self):
        def f(x):
            match x:
                case {"foo": y} if y >= 0:
                    return True
                case {"foo": y} if y < 0:
                    return False

        self.assertIs(f({"foo": 1}), True)
        self.assertIs(f({"foo": -1}), False)

    def test_patma_251(self):
        def f(v, x):
            match v:
                case x.attr if x.attr >= 0:
                    return True
                case x.attr if x.attr < 0:
                    return False
                case _:
                    return None

        class X:
            def __init__(self, attr):
                self.attr = attr

        self.assertIs(f(1, X(1)), True)
        self.assertIs(f(-1, X(-1)), False)
        self.assertIs(f(1, X(-1)), None)

    def test_patma_252(self):
        # Side effects must be possible in guards:
        effects = []
        def lt(x, y):
            effects.append((x, y))
            return x < y

        res = None
        match {"foo": 1}:
            case {"foo": x} if lt(x, 0):
                res = 0
            case {"foo": x} if lt(x, 1):
                res = 1
            case {"foo": x} if lt(x, 2):
                res = 2

        self.assertEqual(res, 2)
        self.assertEqual(effects, [(1, 0), (1, 1), (1, 2)])

    def test_patma_253(self):
        def f(v):
            match v:
                case [x] | x:
                    return x

        self.assertEqual(f(1), 1)
        self.assertEqual(f([1]), 1)

    def test_patma_254(self):
        def f(v):
            match v:
                case {"x": x} | x:
                    return x

        self.assertEqual(f(1), 1)
        self.assertEqual(f({"x": 1}), 1)

    def test_patma_255(self):
        x = []
        match x:
            case [] as z if z.append(None):
                y = 0
            case [None]:
                y = 1
        self.assertEqual(x, [None])
        self.assertEqual(y, 1)
        self.assertIs(z, x)

    def test_patma_runtime_checkable_protocol(self):
        # Runtime-checkable protocol
        from typing import Protocol, runtime_checkable

        @runtime_checkable
        class P(Protocol):
            x: int
            y: int

        class A:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        class B(A): ...

        for cls in (A, B):
            with self.subTest(cls=cls.__name__):
                inst = cls(1, 2)
                w = 0
                match inst:
                    case P() as p:
                        self.assertIsInstance(p, cls)
                        self.assertEqual(p.x, 1)
                        self.assertEqual(p.y, 2)
                        w = 1
                self.assertEqual(w, 1)

                q = 0
                match inst:
                    case P(x=x, y=y):
                        self.assertEqual(x, 1)
                        self.assertEqual(y, 2)
                        q = 1
                self.assertEqual(q, 1)


    def test_patma_generic_protocol(self):
        # Runtime-checkable generic protocol
        from typing import Generic, TypeVar, Protocol, runtime_checkable

        T = TypeVar('T')  # not using PEP695 to be able to backport changes

        @runtime_checkable
        class P(Protocol[T]):
            a: T
            b: T

        class A:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        class G(Generic[T]):
            def __init__(self, x: T, y: T):
                self.x = x
                self.y = y

        for cls in (A, G):
            with self.subTest(cls=cls.__name__):
                inst = cls(1, 2)
                w = 0
                match inst:
                    case P():
                        w = 1
                self.assertEqual(w, 0)

    def test_patma_protocol_with_match_args(self):
        # Runtime-checkable protocol with `__match_args__`
        from typing import Protocol, runtime_checkable

        # Used to fail before
        # https://github.com/python/cpython/issues/110682
        @runtime_checkable
        class P(Protocol):
            __match_args__ = ('x', 'y')
            x: int
            y: int

        class A:
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        class B(A): ...

        for cls in (A, B):
            with self.subTest(cls=cls.__name__):
                inst = cls(1, 2)
                w = 0
                match inst:
                    case P() as p:
                        self.assertIsInstance(p, cls)
                        self.assertEqual(p.x, 1)
                        self.assertEqual(p.y, 2)
                        w = 1
                self.assertEqual(w, 1)

                q = 0
                match inst:
                    case P(x=x, y=y):
                        self.assertEqual(x, 1)
                        self.assertEqual(y, 2)
                        q = 1
                self.assertEqual(q, 1)

                j = 0
                match inst:
                    case P(x=1, y=2):
                        j = 1
                self.assertEqual(j, 1)

                g = 0
                match inst:
                    case P(x, y):
                        self.assertEqual(x, 1)
                        self.assertEqual(y, 2)
                        g = 1
                self.assertEqual(g, 1)

                h = 0
                match inst:
                    case P(1, 2):
                        h = 1
                self.assertEqual(h, 1)


class TestSyntaxErrors(unittest.TestCase):

    def assert_syntax_error(self, code: str):
        with self.assertRaises(SyntaxError):
            compile(inspect.cleandoc(code), "<test>", "exec")

    def test_alternative_patterns_bind_different_names_0(self):
        self.assert_syntax_error("""
        match ...:
            case "a" | a:
                pass
        """)

    def test_alternative_patterns_bind_different_names_1(self):
        self.assert_syntax_error("""
        match ...:
            case [a, [b] | [c] | [d]]:
                pass
        """)


    def test_attribute_name_repeated_in_class_pattern(self):
        self.assert_syntax_error("""
        match ...:
            case Class(a=_, a=_):
                pass
        """)

    def test_imaginary_number_required_in_complex_literal_0(self):
        self.assert_syntax_error("""
        match ...:
            case 0+0:
                pass
        """)

    def test_imaginary_number_required_in_complex_literal_1(self):
        self.assert_syntax_error("""
        match ...:
            case {0+0: _}:
                pass
        """)

    def test_invalid_syntax_0(self):
        self.assert_syntax_error("""
        match ...:
            case {**rest, "key": value}:
                pass
        """)

    def test_invalid_syntax_1(self):
        self.assert_syntax_error("""
        match ...:
            case {"first": first, **rest, "last": last}:
                pass
        """)

    def test_invalid_syntax_2(self):
        self.assert_syntax_error("""
        match ...:
            case {**_}:
                pass
        """)

    def test_invalid_syntax_3(self):
        self.assert_syntax_error("""
        match ...:
            case 42 as _:
                pass
        """)

    def test_len1_tuple_sequence_pattern_comma(self):
        # correct syntax would be `case(*x,):`
        self.assert_syntax_error("""
        match ...:
            case (*x):
                pass
        """)

    def test_mapping_pattern_keys_may_only_match_literals_and_attribute_lookups(self):
        self.assert_syntax_error("""
        match ...:
            case {f"": _}:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_0(self):
        self.assert_syntax_error("""
        match ...:
            case a, a:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_1(self):
        self.assert_syntax_error("""
        match ...:
            case {"k": a, "l": a}:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_2(self):
        self.assert_syntax_error("""
        match ...:
            case MyClass(x, x):
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_3(self):
        self.assert_syntax_error("""
        match ...:
            case MyClass(x=x, y=x):
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_4(self):
        self.assert_syntax_error("""
        match ...:
            case MyClass(x, y=x):
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_5(self):
        self.assert_syntax_error("""
        match ...:
            case a as a:
                pass
        """)

    def test_multiple_assignments_to_name_in_pattern_6(self):
        self.assert_syntax_error("""
        match ...:
            case a as a + 1:  # NAME and expression with no ()
                pass
        """)

    def test_multiple_starred_names_in_sequence_pattern_0(self):
        self.assert_syntax_error("""
        match ...:
            case *a, b, *c, d, *e:
                pass
        """)

    def test_multiple_starred_names_in_sequence_pattern_1(self):
        self.assert_syntax_error("""
        match ...:
            case a, *b, c, *d, e:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_0(self):
        self.assert_syntax_error("""
        match ...:
            case a | "a":
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_1(self):
        self.assert_syntax_error("""
        match 42:
            case x:
                pass
            case y:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_2(self):
        self.assert_syntax_error("""
        match ...:
            case x | [_ as x] if x:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_3(self):
        self.assert_syntax_error("""
        match ...:
            case x:
                pass
            case [x] if x:
                pass
        """)

    def test_name_capture_makes_remaining_patterns_unreachable_4(self):
        self.assert_syntax_error("""
        match ...:
            case x:
                pass
            case _:
                pass
        """)

    def test_patterns_may_only_match_literals_and_attribute_lookups_0(self):
        self.assert_syntax_error("""
        match ...:
            case f"":
                pass
        """)

    def test_patterns_may_only_match_literals_and_attribute_lookups_1(self):
        self.assert_syntax_error("""
        match ...:
            case f"{x}":
                pass
        """)

    def test_real_number_required_in_complex_literal_0(self):
        self.assert_syntax_error("""
        match ...:
            case 0j+0:
                pass
        """)

    def test_real_number_required_in_complex_literal_1(self):
        self.assert_syntax_error("""
        match ...:
            case 0j+0j:
                pass
        """)

    def test_real_number_required_in_complex_literal_2(self):
        self.assert_syntax_error("""
        match ...:
            case {0j+0: _}:
                pass
        """)

    def test_real_number_required_in_complex_literal_3(self):
        self.assert_syntax_error("""
        match ...:
            case {0j+0j: _}:
                pass
        """)

    def test_real_number_multiple_ops(self):
        self.assert_syntax_error("""
        match ...:
            case 0 + 0j + 0:
                pass
        """)

    def test_real_number_wrong_ops(self):
        for op in ["*", "/", "@", "**", "%", "//"]:
            with self.subTest(op=op):
                self.assert_syntax_error(f"""
                match ...:
                    case 0 {op} 0j:
                        pass
                """)
                self.assert_syntax_error(f"""
                match ...:
                    case 0j {op} 0:
                        pass
                """)
                self.assert_syntax_error(f"""
                match ...:
                    case -0j {op} 0:
                        pass
                """)
                self.assert_syntax_error(f"""
                match ...:
                    case 0j {op} -0:
                        pass
                """)

    def test_wildcard_makes_remaining_patterns_unreachable_0(self):
        self.assert_syntax_error("""
        match ...:
            case _ | _:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_1(self):
        self.assert_syntax_error("""
        match ...:
            case (_ as x) | [x]:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_2(self):
        self.assert_syntax_error("""
        match ...:
            case _ | _ if condition():
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_3(self):
        self.assert_syntax_error("""
        match ...:
            case _:
                pass
            case None:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_4(self):
        self.assert_syntax_error("""
        match ...:
            case (None | _) | _:
                pass
        """)

    def test_wildcard_makes_remaining_patterns_unreachable_5(self):
        self.assert_syntax_error("""
        match ...:
            case _ | (True | False):
                pass
        """)

    def test_mapping_pattern_duplicate_key(self):
        self.assert_syntax_error("""
        match ...:
            case {"a": _, "a": _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case0(self):
        self.assert_syntax_error("""
        match ...:
            case {0: _, False: _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case1(self):
        self.assert_syntax_error("""
        match ...:
            case {0: _, 0.0: _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case2(self):
        self.assert_syntax_error("""
        match ...:
            case {0: _, -0: _}:
                pass
        """)

    def test_mapping_pattern_duplicate_key_edge_case3(self):
        self.assert_syntax_error("""
        match ...:
            case {0: _, 0j: _}:
                pass
        """)

class TestTypeErrors(unittest.TestCase):

    def test_accepts_positional_subpatterns_0(self):
        class Class:
            __match_args__ = ()
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_accepts_positional_subpatterns_1(self):
        x = range(10)
        y = None
        with self.assertRaises(TypeError):
            match x:
                case range(10):
                    y = 0
        self.assertEqual(x, range(10))
        self.assertIs(y, None)

    def test_got_multiple_subpatterns_for_attribute_0(self):
        class Class:
            __match_args__ = ("a", "a")
            a = None
        x = Class()
        w = y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y, z):
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_got_multiple_subpatterns_for_attribute_1(self):
        class Class:
            __match_args__ = ("a",)
            a = None
        x = Class()
        w = y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y, a=z):
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_elements_must_be_strings(self):
        class Class:
            __match_args__ = (None,)
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_must_be_a_tuple_0(self):
        class Class:
            __match_args__ = None
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_must_be_a_tuple_1(self):
        class Class:
            __match_args__ = "XYZ"
        x = Class()
        y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y):
                    z = 0
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_match_args_must_be_a_tuple_2(self):
        class Class:
            __match_args__ = ["spam", "eggs"]
            spam = 0
            eggs = 1
        x = Class()
        w = y = z = None
        with self.assertRaises(TypeError):
            match x:
                case Class(y, z):
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

    def test_class_pattern_not_type(self):
        w = None
        with self.assertRaises(TypeError):
            match 1:
                case max(0, 1):
                    w = 0
        self.assertIsNone(w)

    def test_regular_protocol(self):
        from typing import Protocol
        class P(Protocol): ...
        msg = (
            'Instance and class checks can only be used '
            'with @runtime_checkable protocols'
        )
        w = None
        with self.assertRaisesRegex(TypeError, msg):
            match 1:
                case P():
                    w = 0
        self.assertIsNone(w)

    def test_positional_patterns_with_regular_protocol(self):
        from typing import Protocol
        class P(Protocol):
            x: int  # no `__match_args__`
            y: int
        class A:
            x = 1
            y = 2
        w = None
        with self.assertRaises(TypeError):
            match A():
                case P(x, y):
                    w = 0
        self.assertIsNone(w)


class TestValueErrors(unittest.TestCase):

    def test_mapping_pattern_checks_duplicate_key_1(self):
        class Keys:
            KEY = "a"
        x = {"a": 0, "b": 1}
        w = y = z = None
        with self.assertRaises(ValueError):
            match x:
                case {Keys.KEY: y, "a": z}:
                    w = 0
        self.assertIs(w, None)
        self.assertIs(y, None)
        self.assertIs(z, None)

class TestSourceLocations(unittest.TestCase):
    def test_jump_threading(self):
        # See gh-123048
        def f():
            x = 0
            v = 1
            match v:
                case 1:
                    if x < 0:
                        x = 1
                case 2:
                    if x < 0:
                        x = 1
            x += 1

        for inst in dis.get_instructions(f):
            if inst.opcode in dis.hasjump:
                self.assertIsNotNone(inst.positions.lineno, "jump without location")

class TestTracing(unittest.TestCase):

    @staticmethod
    def _trace(func, *args, **kwargs):
        actual_linenos = []

        def trace(frame, event, arg):
            if event == "line" and frame.f_code.co_name == func.__name__:
                assert arg is None
                relative_lineno = frame.f_lineno - func.__code__.co_firstlineno
                actual_linenos.append(relative_lineno)
            return trace

        old_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            func(*args, **kwargs)
        finally:
            sys.settrace(old_trace)
        return actual_linenos

    def test_default_wildcard(self):
        def f(command):                                         # 0
            match command.split():                              # 1
                case ["go", direction] if direction in "nesw":  # 2
                    return f"go {direction}"                    # 3
                case ["go", _]:                                 # 4
                    return "no go"                              # 5
                case _:                                         # 6
                    return "default"                            # 7

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 4, 5])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 4, 6, 7])

    def test_default_capture(self):
        def f(command):                                         # 0
            match command.split():                              # 1
                case ["go", direction] if direction in "nesw":  # 2
                    return f"go {direction}"                    # 3
                case ["go", _]:                                 # 4
                    return "no go"                              # 5
                case x:                                         # 6
                    return x                                    # 7

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 4, 5])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 4, 6, 7])

    def test_no_default(self):
        def f(command):                                         # 0
            match command.split():                              # 1
                case ["go", direction] if direction in "nesw":  # 2
                    return f"go {direction}"                    # 3
                case ["go", _]:                                 # 4
                    return "no go"                              # 5

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 4, 5])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 4])

    def test_only_default_wildcard(self):
        def f(command):               # 0
            match command.split():    # 1
                case _:               # 2
                    return "default"  # 3

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 3])

    def test_only_default_capture(self):
        def f(command):             # 0
            match command.split():  # 1
                case x:             # 2
                    return x        # 3

        self.assertListEqual(self._trace(f, "go n"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "go x"), [1, 2, 3])
        self.assertListEqual(self._trace(f, "spam"), [1, 2, 3])

    def test_unreachable_code(self):
        def f(command):               # 0
            match command:            # 1
                case 1:               # 2
                    if False:         # 3
                        return 1      # 4
                case _:               # 5
                    if False:         # 6
                        return 0      # 7

        self.assertListEqual(self._trace(f, 1), [1, 2, 3])
        self.assertListEqual(self._trace(f, 0), [1, 2, 5, 6])

    #@support.skip_wasi_stack_overflow()
    def test_parser_deeply_nested_patterns(self):
        # Deeply nested patterns can cause exponential backtracking when parsing.
        # See gh-93671 for more information.

        levels = 100

        patterns = [
            "A" + "(" * levels + ")" * levels,
            "{1:" * levels + "1" + "}" * levels,
            "[" * levels + "1" + "]" * levels,
        ]

        for pattern in patterns:
            with self.subTest(pattern):
                code = inspect.cleandoc("""
                    match None:
                        case {}:
                            pass
                """.format(pattern))
                compile(code, "<string>", "exec")


from torch.testing._internal.common_utils import run_tests

if __name__ == "__main__":
    run_tests()
