Environment variables control the disabling of certain explicit SIMD optimizations.

```
x64 options:
TH_NO_AVX2=1 # disable AVX2 codepaths
TH_NO_AVX=1  # disable AVX codepaths
TH_NO_SSE=1  # disable SSE codepaths

ppc64le options:
TH_NO_VSX=1  # disable VSX codepaths
```
