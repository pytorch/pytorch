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

    def test_embedding(self):
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)

            def forward(self, indices):
                return self.emb(indices)

        model = M().eval()
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])

        quantized_node = ns.call_module(nnq.Embedding)
        configs = [
            (float_qparams_weight_only_qconfig, ns.call_module(nnq.Embedding)),
            (None, ns.call_module(nn.Embedding)),
            (default_qconfig, ns.call_module(nn.Embedding)),
        ]

        for qconfig, node in configs:
            qconfig_dict = {"": qconfig}
            m = prepare_fx(model, qconfig_dict)
            self.checkGraphModuleNodes(
                m,
                expected_node_occurrence={
                    ns.call_module(torch.quantization.MinMaxObserver): 0
                },
            )
            m = convert_fx(m)
            self.checkGraphModuleNodes(m, expected_node=node)
            # make sure it runs
            self._compare_script_and_mobile(m, input=indices)


if __name__ == "__main__":
    run_tests()
