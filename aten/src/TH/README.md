Environment variables control the disabling of certain explicit SIMD optimizations.

```
x64 options:
ATEN_CPU_CAPABILITY=avx2    # Force AVX2 codepaths to be used
ATEN_CPU_CAPABILITY=avx     # Force AVX codepaths to be used
ATEN_CPU_CAPABILITY=default # Use oldest supported vector instruction set

ppc64le options:
TH_NO_VSX=1  # disable VSX codepaths
```
