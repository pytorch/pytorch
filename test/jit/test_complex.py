import torch
import os
import sys
from torch.testing._internal.jit_utils import JitTestCase
from typing import List, Dict

# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

class TestComplex(JitTestCase):
    def test_script(self):
        def fn(a: complex):
            return a

        self.checkScript(fn, (3 + 5j,))

    def test_complexlist(self):
        def fn(a: List[complex], idx: int):
            return a[idx]

        input = [1j, 2, 3 + 4j, -5, -7j]
        self.checkScript(fn, (input, 2))

    def test_complexdict(self):
        def fn(a: Dict[complex, complex], key: complex) -> complex:
            return a[key]

        input = {2 + 3j : 2 - 3j, -4.3 - 2j: 3j}
        self.checkScript(fn, (input, -4.3 - 2j))

    def test_pickle(self):
        class ComplexModule(torch.jit.ScriptModule):
            def __init__(self):
                super().__init__()
                # initialization is done in python
                # JIT doesn't parse init
                self.a = 3 + 5j
                self.b = [2 + 3j, 3 + 4j, 0 - 3j, -4 + 0j]
                self.c = {2 + 3j : 2 - 3j, -4.3 - 2j: 3j}

            def forward(self, b: int):
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])
        self.assertEqual(loaded.c, {2 + 3j : 2 - 3j, -4.3 - 2j: 3j})

    def test_complex_parse(self):
        # write more tests for complex(int, int), complex(int, float),
        # complex(float, float), complex(str), complex(float, int)
        # def fn1(a: int):
        #     return a + complex(-2, 3.4)

        def fn(a: int):
            # uses buildConstant
            # we construct python AST
            # Python AST -> JIT AST (JIT IR)
            return a + 5j + 2 + 7.4j - 4

        t = torch.tensor((1,))
        scripted = torch.jit.script(fn)
        self.assertEqual(scripted(t), fn(t))

        # scripted1 = torch.jit.script(fn1)
        # self.assertEqual(scripted1(t), fn1(t))
