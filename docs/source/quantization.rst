.. _quantization-doc:

Quantization
============

We are cetralizing all quantization related development to `torchao <https://github.com/pytorch/ao>`__, please checkout our new doc page: https://docs.pytorch.org/ao/stable/index.html

Plan for the existing quantization flows:
1. Eager mode quantization (torch.ao.quantization.quantize,
torch.ao.quantization.quantize_dynamic), please migrate to use torchao eager mode
`quantize_ <https://docs.pytorch.org/ao/main/generated/torchao.quantization.quantize_.html#torchao.quantization.quantize_>`__ API instead

2. FX graph mode quantization (torch.ao.quantization.quantize_fx.prepare_fx
torch.ao.quantization.quantize_fx.convert_fx, please migrate to use torchao pt2e quantization
API instead (`torchao.quantization.pt2e.quantize_pt2e.prepare_pt2e`, `torchao.quantization.pt2e.quantize_pt2e.convert_pt2e`)

3. pt2e quantization has been migrated to torchao (https://github.com/pytorch/ao/tree/main/torchao/quantization/pt2e)
see https://github.com/pytorch/ao/issues/2259 for more details

We plan to delete `torch.ao.quantization` in 2.10 if there are no blockers, or the earliest PyTorch version until the blockers are cleared.

