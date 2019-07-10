r"""Importing this file includes common utility methods and base clases for
checking quantization api and properties of resulting modules.
"""
import torch
import torch.nn.quantized as nnq
from common_utils import TestCase

# QuantizationTestCase used as a base class for testing quantization on modules
class QuantizationTestCase(TestCase):

    def checkNoPrepModules(self, module):
        r"""Checks the module does not contain child
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertFalse(hasattr(module, 'quant'))
        self.assertFalse(hasattr(module, 'dequant'))

    def checkHasPrepModules(self, module):
        r"""Checks the module contains child
            modules for quantization prepration, e.g.
            quant, dequant and observer
        """
        self.assertTrue(hasattr(module, 'module'))
        self.assertTrue(hasattr(module, 'quant'))
        self.assertTrue(hasattr(module, 'dequant'))

    def checkObservers(self, module):
        r"""Checks the module or module's leaf descendants
            have observers in preperation for quantization
        """
        if hasattr(module, 'qconfig') and module.qconfig is not None and len(module._modules) == 0:
            self.assertTrue(hasattr(module, 'observer'))
        for child in module.children():
            self.checkObservers(child)

    def checkQuantDequant(self, mod):
        r"""Checks that mod has nn.Quantize and
            nn.DeQuantize submodules inserted
        """
        self.assertEqual(type(mod.quant), nnq.Quantize)
        self.assertEqual(type(mod.dequant), nnq.DeQuantize)

    def checkQuantizedLinear(self, mod):
        r"""Checks that mod has been swapped for an nnq.Linear
            module, the bias is qint32, and that the module
            has Quantize and DeQuantize submodules
        """
        self.assertEqual(type(mod.module), nnq.Linear)
        self.assertEqual(mod.module.bias.dtype, torch.qint32)
        self.checkQuantDequant(mod)

    def checkLinear(self, mod):
        self.assertEqual(type(mod), torch.nn.Linear)
