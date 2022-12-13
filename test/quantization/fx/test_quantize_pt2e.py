import torch
import torch._dynamo as torchdynamo
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    skip_if_no_torchvision,
)
from torch.ao.quantization import (
    get_default_qconfig,
    QConfigMapping,
)
from torch.ao.quantization.backend_config import (
    get_qnnpack_backend_config,
)
from torch.ao.quantization.backend_config._pt2e import get_pt2e_backend_config
from torch.ao.quantization.quantize_fx import prepare_fx, convert_to_reference_fx
from torch.ao.quantization._quantize_pt2e import prepare_pt2e, convert_pt2e
from torch.ao.ns.fx.utils import (
    compute_sqnr,
)

class TestQuantizePT2EModels(QuantizationTestCase):
    @skip_if_no_torchvision
    def test_resnet18(self):
        import copy
        import torchvision
        previous = torch.backends.quantized.engine
        torch.backends.quantized.engine = "qnnpack"
        example_inputs = (torch.randn(1, 3, 224, 224),)
        m = torchvision.models.resnet18().eval()
        m_copy = copy.deepcopy(m)
        # long term path, first capture the graph
        # TODO: get official api from Han and Zhengxu
        # TODO: enable functionlization when it's supported in pt2 mode
        # Exception: Invoking operators with non-Fake Tensor inputs in FakeTensorMode is not yet supported. Please convert all Tensors to FakeTensors first. Found in aten.convolution.default(*(FakeTensor(FakeTensor(..., device='meta', size=(1, s0, s1, s1)), cpu)
        # capture_config = CaptureConfig(pt2_mode=True, enable_functionalization=True)
        gm, guards = torchdynamo.export(
            m,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
            tracing_mode="real",
        )

        # this is not available yet
        # infer_nn_stack_trace_and_append_on_mode(m, gm, example_inputs)
        m = gm
        print("after exir.capture:", m)

        # TODO: define qconfig_mapping specifically for executorch
        backend_config = get_pt2e_backend_config()
        qconfig = get_default_qconfig("qnnpack")
        qconfig_mapping = QConfigMapping().set_global(qconfig)
        print("m: ", m)
        # TODO: check qconfig_mapping to make sure conv and bn are both configured
        # to be quantized before fusion
        # TODO: check correctness for conv bn fusion
        before_fusion_result = m(*example_inputs)
        before_quant_result = before_fusion_result
        # _fuse_conv_bn_(m)
        after_fusion_result = m(*example_inputs)
        print(before_fusion_result - after_fusion_result)
        self.assertTrue(torch.max(before_fusion_result - after_fusion_result) < 1e-5)
        print("fused:", m)
        m = prepare_pt2e(m, qconfig_mapping, example_inputs, backend_config)
        print("after prepare:", m)
        self.assertEqual(id(m.activation_post_process_3), id(m.activation_post_process_2))
        after_prepare_result = m(*example_inputs)
        m = convert_pt2e(m)
        print("after convert:", m)
        # TODO: conv, conv_relu, linear delegation
        # quantized ops to implement: add_relu
        # compile_config = EdgeCompileConfig(passes=[QuantFusionPass(), SpecPropPass()])
        # m = exir.capture(m, example_inputs, config=capture_config).to_edge(config=compile_config)
        # print("after lowering:", m)
        after_quant_result = m(*example_inputs)
        # # check
        # print("v.s. fp32 diff:", torch.max(before_quant_result - after_quant_result))
        # print("v.s. fp32 sqnr:", compute_sqnr(before_quant_result, after_quant_result))
        # TODO: re-enable after we can turn on functionalization again
        # m = m.to_executorch()

        # comparing with existing fx graph mode quantization reference flow
        # backend_config = get_qnnpack_backend_config()
        m_fx = prepare_fx(m_copy, qconfig_mapping, example_inputs) # backend_config=backend_config)
        after_prepare_result_fx = m_fx(*example_inputs)
        m_fx = convert_to_reference_fx(m_fx)#, backend_config=backend_config)
        print("fx reference result:", m_fx)
        after_quant_result_fx = m_fx(*example_inputs)
        print("after prepare:")
        print("v.s. fx diff:", torch.max(after_prepare_result - after_prepare_result_fx))
        print("v.s. fx sqnr:", compute_sqnr(after_prepare_result, after_prepare_result_fx))
        print("after convert:")
        print("v.s. fx diff:", torch.max(after_quant_result - after_quant_result_fx))
        print("v.s. fx sqnr:", compute_sqnr(after_quant_result, after_quant_result_fx))
        print("conv1 input:", m_fx.conv1_input_scale_0, m_fx.conv1_input_zero_point_0)
        print("conv1 output:", m_fx.conv1_scale_0, m_fx.conv1_zero_point_0)
        print("fc output:", m_fx.fc_scale_0, m_fx.fc_zero_point_0)
        torch.backends.quantized.engine = previous
        # m = m.to_executorch()
        # m(*example_inputs)
