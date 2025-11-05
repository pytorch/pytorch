import re

import torch


# It is not a good idea to directly apply hipify_torch to codegen, which will be vulnerable to cases like:
#   "...
#    from ..codecache import CudaKernelParamCache
#   ..."
# In such cases, we do not need to hipify_torch the original class/file name in codegen/codecache


def maybe_hipify_code_wrapper(source_codes: str, force_hipify: bool = False) -> str:
    if torch.version.hip is None and not force_hipify:
        return source_codes

    try:
        from torch.utils.hipify.hipify_python import PYTORCH_MAP, PYTORCH_TRIE
    except ImportError:
        # hipify not available for non-AMD builds
        return source_codes

    def c2_repl(m: re.Match[str]) -> object:
        return PYTORCH_MAP[m.group(0)]

    # We need to redefine RE_PYTORCH_PREPROCESSOR here since in hipify_torch,
    # it will apply positive lookbehind (?<=\W) to the pattern to avoid matching
    # keyword at the beginning of code line. However, this can happen in codegen,
    # which will cause the pattern to not match.

    # Note that lookahead (?=\W) is still needed to keep hipification idomponent, for example
    # we need to skip replacing "getStreamFromExternal" in "getStreamFromExternalMasqueradingAsCUDA"
    RE_PYTORCH_PREPROCESSOR = re.compile(rf"({PYTORCH_TRIE.export_to_regex()})(?=\W)")

    source_codes = RE_PYTORCH_PREPROCESSOR.sub(c2_repl, source_codes)  # type: ignore[arg-type]
    return source_codes
