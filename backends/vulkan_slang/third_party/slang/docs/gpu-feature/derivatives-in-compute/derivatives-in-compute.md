### Derivatives In Compute 
An entry point may be decorated with `[DerivativeGroupQuad]` or `[DerivativeGroupLinear]` to specify how to use derivatives in compute shaders.

GLSL syntax may also be used, but is not recommended (`derivative_group_quadsNV`/`derivative_group_linearNV`).

Targets:
* **_SPIRV:_** Enables `DerivativeGroupQuadsNV` or `DerivativeGroupLinearNV`.
* **_GLSL:_** Enables `derivative_group_quadsNV` or `derivative_group_LinearNV`.
* **_HLSL:_** Does nothing. `sm_6_6` is required to use derivatives in compute shaders. HLSL uses an equivalent of `DerivativeGroupQuad`.
