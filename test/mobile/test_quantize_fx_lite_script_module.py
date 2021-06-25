import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.utils.bundled_inputs

# graph mode quantization based on fx
from torch.quantization.quantize_fx import (
    prepare_fx,
    convert_fx,
)
from torch.testing._internal.common_quantization import NodeSpec as ns
from torch.testing._internal.common_quantization import QuantizationLiteTestCase
from torch.quantization import (
    default_qconfig,
    float_qparams_weight_only_qconfig,
)


class TestFuseFx(QuantizationLiteTestCase):

    # Tests from:
    # ./caffe2/test/quantization/fx/test_quantize_fx.py

    def test_embedding(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        model = M().eval()
        indices = torch.randint(low=0, high=10, size=(20,))

        quantized_node = ns.call_module(nnq.Embedding)
        configs = [
            (float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)),
            (None, ns.call_module(nn.Embedding)),
            (default_qconfig, ns.call_module(nn.Embedding)),
        ]

        for qconfig, node in configs:
            qconfig_dict = {"": qconfig}
            m = prepare_fx(model, qconfig_dict)
            m = convert_fx(m)
            self._compare_script_and_mobile(m, input=indices)

    def test_conv2d(self):
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.conv1 = nn.Conv2d(1, 1, 1)
                self.conv2 = nn.Conv2d(1, 1, 1)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                return x

        m = M().eval()
        qconfig_dict = {"object_type": [(torch.nn.Conv2d, default_qconfig)]}
        m = prepare_fx(m, qconfig_dict)
        data = torch.randn(1, 1, 1, 1)
        m = convert_fx(m)
        # first conv is quantized, second conv is not quantized
        node_list = [
            ns.call_function(torch.quantize_per_tensor),
            ns.call_module(nnq.Conv2d),
            ns.call_module(nnq.Conv2d),
            ns.call_method("dequantize"),
        ]
        self._compare_script_and_mobile(m, input=data)


if __name__ == "__main__":
    run_tests()
