import torch

from torch.utils.hipify.hipify_python import PYTORCH_MAP, RE_PYTORCH_PREPROCESSOR

# It is not a good idea to directly apply hipify_torch to codegen, which will be vulnerable to cases like:
#   "...
#    from ..codecache import CudaKernelParamCache
#   ..."
# In such cases, we do not need to hipify_torch the orignial class/file name in codegen/codecache


def maybe_hipify_code_wrapper(source_codes: str) -> str:
    if torch.version.hip is None:
        return source_codes

    def c2_repl(m):
        return PYTORCH_MAP[m.group(0)]

    source_codes = RE_PYTORCH_PREPROCESSOR.sub(c2_repl, source_codes)
    return source_codes
