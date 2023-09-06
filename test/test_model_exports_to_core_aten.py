# Owner(s): ["oncall: mobile"]
import copy

import torch
import torch._export as export
from torch._export.verifier import ATenDialectVerifier, SpecViolationError

from torch.testing._internal.common_quantization import skip_if_no_torchvision
from torch.testing._internal.common_utils import TestCase


class TestPT2ETModels(TestCase):
    def _check_core_aten_ops(self, gm: torch.fx.GraphModule):
        """
        Check if all ops in the graph module are core aten ops.
        """
        verifier = ATenDialectVerifier()
        for node in gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target not in verifier.valid_builtin_funcs()
            ):
                with self.assertRaises(SpecViolationError):
                    verifier.check_valid_op(node.target)

    @skip_if_no_torchvision
    def test_vit_export_to_core_aten(self):
        from torchvision.models import vit_b_16  # @manual

        m = vit_b_16(weights="IMAGENET1K_V1")
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        m = export.capture_pre_autograd_graph(m, copy.deepcopy(example_inputs))
        m(*example_inputs)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    @skip_if_no_torchvision
    def test_mv2_export_to_core_aten(self):
        from torchvision.models import mobilenet_v2  # @manual

        m = mobilenet_v2(pretrained=True)
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    @skip_if_no_torchvision
    def test_mv3_export_to_core_aten(self):
        from torchvision.models import mobilenet_v3_small  # @manual

        m = mobilenet_v3_small(pretrained=True)
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    @skip_if_no_torchvision
    def test_inception_v3_export_to_core_aten(self):
        from torchvision.models import inception_v3  # @manual

        m = inception_v3(weights="IMAGENET1K_V1")
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    @skip_if_no_torchvision
    def test_resnet18_export_to_core_aten(self):
        from torchvision.models import resnet18, ResNet18_Weights  # @manual

        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    @skip_if_no_torchvision
    def test_resnet50_export_to_core_aten(self):
        from torchvision.models import resnet50, ResNet50_Weights  # @manual

        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        m = m.eval()
        input_shape = (1, 3, 224, 224)
        example_inputs = (torch.randn(input_shape),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    def test_wav2letter_export_to_core_aten(self):
        from torchaudio import models

        batch_size = 10
        input_frames = 700
        vocab_size = 4096

        m = models.Wav2Letter(num_classes=vocab_size)
        m = m.eval()
        input_shape = (batch_size, 1, input_frames)
        example_inputs = (torch.randn(input_shape),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    def test_inception_v4_export_to_core_aten(self):
        from timm.models import inception_v4

        m = inception_v4(pretrained=True)
        m = m.eval()
        input_shape = (3, 299, 299)
        example_inputs = (torch.randn(input_shape).unsqueeze(0),)
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)

    def test_mobilebert_export_to_core_aten(self):
        from transformers import AutoTokenizer, MobileBertModel  # @manual

        m = MobileBertModel.from_pretrained(
            "google/mobilebert-uncased", return_dict=False
        )
        m = m.eval()
        tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")
        example_inputs = (
            tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"],
        )
        gm = export.export(m, copy.deepcopy(example_inputs))
        self._check_core_aten_ops(gm)
