from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.nn.quantized as nnq
from torch.quantization import default_eval_fn, QConfig, default_qconfig, \
    default_observer, quantize, prepare, convert

from common_utils import run_tests
from common_quantization import QuantizationTestCase, SingleLayerLinearModel, \
    TwoLayerLinearModel, NestedModel, WrappedModel, ManualQuantModel


calib_data = [torch.rand(20, 5, dtype=torch.float) for _ in range(20)]

class ModelQuantizeAPITest(QuantizationTestCase):

    def test_single_layer(self):
        r"""Quantize SingleLayerLinearModel which has one Linear module, make sure it is swapped
        to nnq.Linear which is the quantized version of the module
        """
        model = SingleLayerLinearModel()
        qconfig_dict = {
            '': default_qconfig
        }
        model = prepare(model, qconfig_dict)
        # Check if observers and quant/dequant nodes are inserted
        self.checkNoPrepModules(model)
        self.checkHasPrepModules(model.fc1)
        self.checkObservers(model)

        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.checkHasPrepModules(model.fc1)
            self.checkQuantizedLinear(model.fc1)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(SingleLayerLinearModel(), default_eval_fn, calib_data, qconfig_dict)
        checkQuantized(model)

    def test_two_layers(self):
        r"""TwoLayerLinearModel has two Linear modules but we only quantize the second one
        `fc2`, and `fc1`is not quantized
        """
        model = TwoLayerLinearModel()
        qconfig_dict = {
            'fc2': default_qconfig
        }
        model = prepare(model, qconfig_dict)

        self.checkNoPrepModules(model)
        self.checkObservers(model)
        self.checkNoPrepModules(model.fc1)
        self.checkHasPrepModules(model.fc2)

        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.fc1)
            self.checkHasPrepModules(model.fc2)
            self.assertEqual(type(model.fc1), torch.nn.Linear)
            self.checkQuantizedLinear(model.fc2)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(TwoLayerLinearModel(), default_eval_fn, calib_data, qconfig_dict)
        checkQuantized(model)

    def test_nested1(self):
        r"""Test quantization for nested model, top level 'fc3' and
        'fc1' of submodule 'sub2', 'sub2.fc2' is not quantized
        """
        model = NestedModel()
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2.fc1': default_qconfig
        }

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkNoPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        model = prepare(model, qconfig_dict)
        checkPrepModules(model, True)
        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            checkPrepModules(model)
            self.checkLinear(model.sub1.fc)
            self.checkQuantizedLinear(model.fc3)
            self.checkQuantizedLinear(model.sub2.fc1)
            self.checkLinear(model.sub2.fc2)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(NestedModel(), default_eval_fn, calib_data, qconfig_dict)
        checkQuantized(model)


    def test_nested2(self):
        r"""Another test case for quantized, we will quantize all submodules
        of submodule sub2, this will include redundant quant/dequant, to
        remove them we need to manually call QuantWrapper or insert
        QuantStub/DeQuantStub, see `test_quant_dequant_wrapper` and
        `test_manual`
        """
        model = NestedModel()
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2': default_qconfig
        }
        model = prepare(model, qconfig_dict)

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkHasPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(model, True)

        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            checkPrepModules(model)
            self.checkLinear(model.sub1.fc)
            self.assertEqual(type(model.sub1.relu), torch.nn.ReLU)
            self.checkQuantizedLinear(model.sub2.fc1)
            self.checkQuantizedLinear(model.sub2.fc2)
            self.checkQuantizedLinear(model.fc3)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(NestedModel(), default_eval_fn, calib_data, qconfig_dict)
        checkQuantized(model)

    def test_nested3(self):
        r"""More complicated nested test case with child qconfig overrides
        parent qconfig
        """
        model = NestedModel()
        custum_options = {
            'dtype': torch.quint8,
            'qscheme': torch.per_tensor_affine
        }
        custom_qconfig = QConfig(weight=default_observer(),
                                 activation=default_observer(**custum_options))
        qconfig_dict = {
            'fc3': default_qconfig,
            'sub2': default_qconfig,
            'sub2.fc1': custom_qconfig
        }
        model = prepare(model, qconfig_dict)

        def checkPrepModules(model, before_calib=False):
            if before_calib:
                self.checkObservers(model)
            self.checkNoPrepModules(model)
            self.checkNoPrepModules(model.sub1)
            self.checkNoPrepModules(model.sub1.fc)
            self.checkNoPrepModules(model.sub1.relu)
            self.checkNoPrepModules(model.sub2)
            self.checkHasPrepModules(model.sub2.fc1)
            self.checkHasPrepModules(model.sub2.fc2)
            self.checkHasPrepModules(model.fc3)

        checkPrepModules(model, True)

        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            checkPrepModules(model)
            self.checkQuantizedLinear(model.sub2.fc1)
            self.checkQuantizedLinear(model.sub2.fc2)
            self.checkQuantizedLinear(model.fc3)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(NestedModel(), default_eval_fn, calib_data, qconfig_dict)
        checkQuantized(model)

    def test_quant_wrapper(self):
        r"""User need to modify the original code with QuantWrapper,
        and call the quantization utility functions.
        """
        model = WrappedModel()

        # since we didn't provide qconfig_dict, the model is modified inplace
        # but we can do `model = prepare(model)` as well
        prepare(model)
        self.checkObservers(model)

        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            self.checkLinear(model.fc)
            self.checkQuantDequant(model.sub)
            self.assertEqual(type(model.sub.module.fc1), nnq.Linear)
            self.assertEqual(type(model.sub.module.fc2), nnq.Linear)
            self.assertEqual(type(model.sub.module.relu), nnq.ReLU)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(WrappedModel(), default_eval_fn, calib_data, {})
        checkQuantized(model)


    def test_manual(self):
        r"""User inserts QuantStub and DeQuantStub in model code
        and call the quantization utility functions.
        """
        model = ManualQuantModel()
        # propagate the qconfig of parents to children, model is changed
        # inplace
        prepare(model)
        self.checkObservers(model)

        default_eval_fn(model, calib_data)
        convert(model)

        def checkQuantized(model):
            self.assertEqual(type(model.fc), nnq.Linear)
            default_eval_fn(model, calib_data)

        checkQuantized(model)

        # test one line API
        model = quantize(ManualQuantModel(), default_eval_fn, calib_data)
        checkQuantized(model)


if __name__ == '__main__':
    run_tests()
