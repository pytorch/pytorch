import torch
from torch._inductor.codegen.rocm.rocm_template import ROCmTemplate
from torch._inductor.ir import IRNode
from torch._inductor.utils import IndentedBuffer


class CKTileTemplate(ROCmTemplate):
    """
    Base class for generating CK templates, has common, i.e. non-gemm-specific, code generation logic
    """

    _TORCH_DTYPE_TO_CK = {
        torch.float32: "F32",
        torch.float64: "F64",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int32: "I32",
        torch.int8: "I8",
        torch.float8_e4m3fnuz: "F8",
        torch.float8_e5m2fnuz: "BF8",
    }

    ck_dtype_to_size = {
        "FP16": 2,
        "BF16": 2,
    }

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
                // CK headers
                #include "ck_tile/core.hpp"

            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
                using F8  = ck_tile::fp8_t;
                using BF8 = ck_tile::bf8_t;
                using F16 = ck_tile::half_t;
                using F32 = float;
                using BF16 = ck_tile::bfloat16_t;
            """
        )
        return res

    def torch_type_to_ck(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._TORCH_DTYPE_TO_CK.get(node.get_dtype())}*)({ptr})"
