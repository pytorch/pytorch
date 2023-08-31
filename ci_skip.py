import collections
from typing import NamedTuple

class CI(NamedTuple):
    backend: str  # aot_eager or inductor
    training: bool
    dynamic: bool = False
    device: str = "cuda"

CI_SKIP = collections.defaultdict(list)


# Skips for dynamic=False

# Here eager really means dynamo+eager
CI_SKIP[CI("eager", training=False)] = [
    # TorchBench
    "DALLE2_pytorch",  # AttributeError: text_encodings
    "hf_BigBird",  # fail_accuracy
    # TypeError: pad_center() takes 1 positional argument but 2 were given
    "tacotron2",
    # Huggingface
    "DebertaV2ForQuestionAnswering",  # OOM
]

CI_SKIP[CI("eager", training=True)] = [
    *CI_SKIP[CI("eager", training=False)],
    # TorchBench
    "BERT_pytorch",  # accuracy
    "Background_Matting",  # fp64_OOM
    "hf_BigBird",  # fp64_OOM
    "hf_T5_base",  # fp64_OOM
    "llama",  # Accuracy failed: allclose not within tol=0.001
    "vision_maskrcnn",  # The size of tensor a (29) must match the size of tensor b (33) (doesn't repro)
    # Huggingface
    "XGLMForCausalLM",  # OOM
    # TIMM
    "cait_m36_384",  # fp64_OOM
    "convit_base",  # fp64_OOM
    "mobilenetv2_100",  # accuracy
    "xcit_large_24_p8_224",  # fp64_OOM,
]

CI_SKIP[CI("aot_eager", training=False)] = [
    *CI_SKIP[CI("eager", training=False)],
    # all dynamic shapes errors for detectron variants
    "demucs",  # OOM
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_fcos_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "hf_BigBird",  # OOM
    "tacotron2",  # AssertionError: Deduped args out of bounds
    # Huggingface
    "BartForConditionalGeneration",  # OOM
    "DebertaV2ForQuestionAnswering",  # OOM
    # Torchbench
    "speech_transformer",  # https://github.com/pytorch/pytorch/issues/99893
    "pyhpc_isoneutral_mixing",  # https://github.com/pytorch/pytorch/issues/99893
    "pyhpc_turbulent_kinetic_energy",  # https://github.com/pytorch/pytorch/issues/99893
]

CI_SKIP[CI("aot_eager", training=True)] = [
    *CI_SKIP[CI("aot_eager", training=False)],
    # TorchBench
    "Background_Matting",  # fp64_OOM
    "hf_T5_base",  # fp64_OOM
    "mobilenet_v2_quantized_qat",  # fp64_OOM
    "resnet50_quantized_qat",  # fp64_OOM
    "pytorch_struct",
    # Huggingface
    "MBartForConditionalGeneration",  # OOM
    "M2M100ForConditionalGeneration",  # OOM
    "XGLMForCausalLM",  # OOM
    # TIMM
    "cait_m36_384",  # fp64_OOM
    "convit_base",  # fp64_OOM
    "fbnetv3_b",  # Accuracy (blocks.2.2.bn1.weight.grad)
    "levit_128",  # Accuracy (patch_embed.0.c.weight.grad)
    "lcnet_050",  # Accuracy (blocks.1.0.bn2.weight.grad)
    "sebotnet33ts_256",  # Accuracy (stem.conv1.conv.weight.grad)
    "xcit_large_24_p8_224",  # fp64_OOM,
]

CI_SKIP[CI("inductor", training=False)] = [
    # TorchBench
    "DALLE2_pytorch",  # AttributeError: text_encodings
    "demucs",  # OOM
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_fcos_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    # TorchBench
    "detectron2",
    "densenet121",  # flaky accuracy
    "hf_T5",  # accuracy
    "hf_BigBird",  # accuracy
    "hf_GPT2_large",  # OOM
    "maml",  # accuracy
    "mobilenet_v2_quantized_qat",  # The eval test only supports CPU
    "pytorch_struct",  # Test eval is not implemented
    "pyhpc_equation_of_state",  # Accuracy
    "pyhpc_turbulent_kinetic_energy",  # Accuracy
    "tacotron2",
]

CI_SKIP[CI("inductor", training=False, device="cpu")] = [
    # TorchBench
    "drq",  # Need to update torchbench
    "detectron2_fasterrcnn_r_101_c4",
    "detectron2_fasterrcnn_r_101_dc5",
    "detectron2_fasterrcnn_r_101_fpn",
    "detectron2_fasterrcnn_r_50_c4",
    "detectron2_fasterrcnn_r_50_dc5",
    "detectron2_fasterrcnn_r_50_fpn",
    "detectron2_fcos_r_50_fpn",
    "detectron2_maskrcnn_r_101_c4",
    "detectron2_maskrcnn_r_101_fpn",
    "detectron2_maskrcnn_r_50_c4",
    "detectron2_maskrcnn_r_50_fpn",
    "doctr_det_predictor",  # requires newer gcc
    "doctr_reco_predictor",  # requires newer gcc
    "gat",  # does not work with fp32
    "gcn",  # does not work with fp32
    "hf_Bert_large",  # OOM
    "hf_GPT2_large",  # Intermittent failure on CI
    "hf_T5_base",  # OOM
    "mobilenet_v2_quantized_qat",
    "pyhpc_turbulent_kinetic_energy",
    "resnet50_quantized_qat",  # Eager model failed to run(Quantize only works on Float Tensor, got Double)
    "sage",  # does not work with fp32
    # Huggingface
    "MBartForConditionalGeneration",  # Accuracy https://github.com/pytorch/pytorch/issues/94793
    "PLBartForConditionalGeneration",  # Accuracy https://github.com/pytorch/pytorch/issues/94794
    # TIMM
    "cait_m36_384",  # Accuracy
    "pnasnet5large",  # OOM
    "xcit_large_24_p8_224",  # OOM https://github.com/pytorch/pytorch/issues/95984
    "opacus_cifar10",  # Fails to run https://github.com/pytorch/pytorch/issues/99201
]

CI_SKIP[CI("inductor", training=True)] = [
    *CI_SKIP[CI("inductor", training=False)],
    # TorchBench
    "Background_Matting",  # fp64_OOM
    "hf_T5_base",  # accuracy
    "mobilenet_v3_large",  # accuracy
    "resnet50_quantized_qat",  # Eager model failed to run
    "AlbertForQuestionAnswering",  # accuracy
    "crossvit_9_240",  # fails to run on timm 0.8.22 with cudagraphs, mempools
    "deit_base_distilled_patch16_224",  # fails to run in timm 0.8.22, cudagraphs
    "mobilevit_s",
    "pit_b_224",
    "twins_pcpvt_base",
    "visformer_small",
    "vit_base_patch16_224",
    "xcit_large_24_p8_224",
]

# Skips for dynamic=True

CI_SKIP[CI("aot_eager", training=False, dynamic=True)] = [
    *CI_SKIP[CI("aot_eager", training=False)],
    "vision_maskrcnn",  # accuracy failure on boxes, after https://github.com/pytorch/pytorch/issues/101093
    # https://github.com/pytorch/pytorch/issues/103760
    "hf_T5_generate",
    "hf_Bert",  # Error: RelaxedUnspecConstraint(L['input_ids'].size()[0]) - inferred constant (4)
]

CI_SKIP[CI("aot_eager", training=True, dynamic=True)] = [
    *CI_SKIP[CI("aot_eager", training=True)],
    *CI_SKIP[CI("aot_eager", training=False, dynamic=True)],
    "llama",  # AssertionError: cannot compute free_symbols of True
    "torchrec_dlrm",  # RuntimeError: mat1 and mat2 must have the same dtype, but got Float and BFloat16
]

CI_SKIP[CI("inductor", training=False, dynamic=True)] = [
    *CI_SKIP[CI("aot_eager", training=False, dynamic=True)],
    *CI_SKIP[CI("inductor", training=False)],
    "nanogpt_generate",  # Assertion `index out of bounds: 0 <= tmp0 < 64` failed.
]

CI_SKIP[CI("inductor", training=True, dynamic=True)] = [
    # NB: Intentionally omitting for symmetry with dynamic=False
    # *CI_SKIP[CI("aot_eager", training=True, dynamic=True)],
    *CI_SKIP[CI("inductor", training=False, dynamic=True)],
    *CI_SKIP[CI("inductor", training=True)],
    "levit_128",  # Accuracy fails on A10G, passes on A100
    "sebotnet33ts_256",  # Flaky accuracy failed
]

CI_SKIP[CI("inductor", training=False, dynamic=True, device="cpu")] = [
    *CI_SKIP[CI("inductor", training=False, device="cpu")],
    "pyhpc_isoneutral_mixing",
    "dpn107",
]

CI_SKIP_OPTIMIZER = {
    # TIMM
    "convmixer_768_32",  # accuracy
    "hrnet_w18",  # Stack issue in fx
    # HF
    "pnasnet5large",  # Stack issue in fx
    "MobileBertForMaskedLM",  # Stack issue in fx
    "MobileBertForQuestionAnswering",  # Stack issue in fx
    "PegasusForConditionalGeneration",  # OOM
}

CI_SKIP_DYNAMIC_BATCH_ONLY = {
    "sam",
    # See https://github.com/mindee/doctr/blob/f2114758d529ed8d3d0030581638f0520b6b98d8/doctr/models/detection/core.py#L89
    # It iterates over the batch, which is dynamic, and dynamo chokes
    # We should be able to graphbreak there.
    "doctr_det_predictor",
    "dlrm",
}

NON_DETERMINISTIC_MODELS = {
    "alexnet",
    "Background_Matting",
    "pytorch_CycleGAN_and_pix2pix",
    "pytorch_unet",
    "Super_SloMo",
    "vgg16",
    # https://github.com/pytorch/pytorch/issues/96724
    "Wav2Vec2ForCTC",
    "Wav2Vec2ForPreTraining",
    "sam",
}
