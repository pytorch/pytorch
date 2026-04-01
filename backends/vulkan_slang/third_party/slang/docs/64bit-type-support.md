Slang 64-bit Type Support
=========================

## Summary

* Not all targets support 64 bit types, or all 64 bit types 
  * 64 bit integers generally require later APIs/shader models
* When specifying 64 bit floating-point literals *always* use the type suffixes (ie `L`) 
* An integer literal will be interpreted as 64 bits if it cannot fit in a 32 bit value.
* GPU target/s generally do not support all double intrinsics 
  * Typically missing are trascendentals (sin, cos etc), logarithm and exponential functions
  * CUDA is the exception supporting nearly all double intrinsics
* D3D 
  * D3D targets *appear* to support double intrinsics (like sin, cos, log etc), but behind the scenes they are actually being converted to float
  * When using D3D12, it is best to use DXIL if you use double because there are some serious issues around double and DXBC
* VK will produce an error in validation if a double intrinsic is used it does support (which is most of them)
* Vector and Matrix types have even spottier than scalar intrinsic support across targets

Overview
========

The Slang language supports 64 bit built in types. Such as

* `double`
* `uint64_t`
* `int64_t`

This also applies to vector and matrix versions of these types. 

Unfortunately if a specific target supports the type or the typical HLSL intrinsic functions (such as sin/cos/max/min etc) depends very much on the target.

Special attention has to be made with respect to literal 64 bit types. By default float literals if they do not have an explicit suffix are assumed to be 32 bit. There is a variety of reasons for this design choice - the main one being around by default behavior of getting good performance. The suffixes required for 64 bit types are as follows

```
// double - 'l' or 'L'

double a = 1.34e-200L;
// WRONG!: This is the same as b = double(float(1.34e-200)) which will be 0. Will produce a warning.
double b = 1.34e-200; 

// int64_t - 'll' or 'LL' (or combination of upper/lower)

int64_t c = -5436365345345234ll;

int64_t e = ~0LL;       // Same as 0xffffffffffffffff

// uint64_t - 'ull' or 'ULL' (or combination of upper/lower)

uint64_t g = 0x8000000000000000ull; 

uint64_t i = ~0ull;       // Same as 0xffffffffffffffff
uint64_t j = ~0;          // Equivalent to 'i' because uint64_t(int64_t(~int32_t(0)));
```

These issues are discussed more on issue [#1185](https://github.com/shader-slang/slang/issues/1185)

The type of a decimal non-suffixed integer literal is the first integer type from the list [`int`, `int64_t`] 
which can represent the specified literal value. If the value cannot fit, the literal is  represented as an `uint64_t` 
and a warning is given.
The type of a hexadecimal non-suffixed integer literal  is the first type from the list [`int`, `uint`, `int64_t`, `uint64_t`] 
that can represent the specified literal value. A non-suffixed integer literal will be 64 bit if it cannot fit in 32 bits.
```
// Same as int64_t a = int(1), the value can fit into a 32 bit integer.
int64_t a = 1;

// Same as int64_t b = int64_t(2147483648), the value cannot fit into a 32 bit integer.
int64_t b = 2147483648;

// Same as int64_t c = uint64_t(18446744073709551615), the value is larger than the maximum value of a signed 64 bit
// integer, and is interpreted as an unsigned 64 bit integer. Warning is given.
uint64_t c = 18446744073709551615;

// Same as uint64_t = int(0x7FFFFFFF), the value can fit into a 32 bit integer.
uint64_t d = 0x7FFFFFFF;

// Same as uint64_t = int64_t(0x7FFFFFFFFFFFFFFF), the value cannot fit into an unsigned 32 bit integer but
// can fit into a signed 64 bit integer.
uint64_t e = 0x7FFFFFFFFFFFFFFF;

// Same as uint64_t = uint64_t(0xFFFFFFFFFFFFFFFF), the value cannot fit into a signed 64 bit integer, and
// is interpreted as an unsigned 64 bit integer.
uint64_t f = 0xFFFFFFFFFFFFFFFF;
```

Double support
==============

Target   | Compiler/Binary  |  Double Type   |   Intrinsics          |  Notes
---------|------------------|----------------|-----------------------|-----------
CPU      |                  |      Yes       |          Yes          |  1
CUDA     | Nvrtx/PTX        |      Yes       |          Yes          |  1
D3D12    | DXC/DXIL         |      Yes       |          Small Subset |  4 
Vulkan   | GlSlang/Spir-V   |      Yes       |          Partial      |  2
D3D11    | FXC/DXBC         |      Yes       |          Small Subset |  4
D3D12    | FXC/DXBC         |      Yes       |          Small Subset |  3, 4

1) CUDA and CPU support most intrinsics, with the notable exception currently of matrix invert
2) In terms of lack of general intrinsic support, the restriction is described in  https://www.khronos.org/registry/spir-v/specs/1.0/GLSL.std.450.html

The following intrinsics are available for Vulkan 

`fmod` (as %), `rcp`, `sign`, `saturate`, `sqrt`, `rsqrt`, `frac`, `ceil`, `floor`, `trunc`, `abs`, `min`, `max`, `smoothstep`, `lerp`, `clamp`, `step` and `asuint`. 

These are tested in the test `tests/hlsl-intrinsic/scalar-double-vk-intrinsic.slang`.

What is missing are transedentals, expX, logX. 

Note that GlSlang does produce Spir-V that contains double intrinsic calls for the missing intrinsics, the failure happens when validating the Spir-V 

```
Validation: error 0:  [ UNASSIGNED-CoreValidation-Shader-InconsistentSpirv ] Object: VK_NULL_HANDLE (Type = 0) | SPIR-V module not valid: GLSL.std.450 Sin: expected Result Type to be a 16 or 32-bit scalar or vector float type
  %57 = OpExtInst %double %1 Sin %56
```

3) That if a RWStructuredBuffer<double> is used on D3D12 with DXBC, and a double is written, it can lead to incorrect behavior. Thus it is recommended not to use double with dxbc, but to use dxil to keep things simple. A test showing this problem is `tests/bugs/dxbc-double-problem.slang`. The test `tests/hlsl-intrinsic/scalar-double-simple.slang` shows not using a double resource, doubles do appear to work on D3D12 DXBC. 

4) If you compile code using double and intrinsics through Slang at first blush it will seem to work. Assuming there are no errors in your code, your code will even typically appear to work correctly. Unfortunately what is really happening is the backend compiler (fxc or dxc) compiler is narrowing double to float and then using float intrinsics. It typically generates a warning when this happens, but unless there is an error in your code you will not see these warnings because dxc doesn't appear to have a mechanism to return warnings if there isn't an error. This is why everything appears to work - but actually any intrinsic call is losing precision silently. 

Note on dxc by default Slang disables warnings - warnings need to be enabled to see the narrowing warnings. 

There is another exception around the use of % - if you do this with double it will return an error saying on float is supported. 

It appears that no intrinsics are available for double with fxc. 

On dxc the following intrinsics are available with double::

`rcp`, `sign`, `saturate`, `abs`, `min`, `max`, `clamp`, `asuint`. 

These are tested in the test `tests/hlsl-intrinsic/scalar-double-d3d-intrinsic.slang`.

There is no support for transcendentals (`sin`, `cos` etc) or `log`/`exp`. More surprising is that `sqrt`, `rsqrt`, `frac`, `ceil`, `floor`, `trunc`, `step`, `lerp`, `smoothstep` are also not supported.

uint64_t and int64_t Support
============================

Target   | Compiler/Binary  | u/int64_t Type |  Intrinsic support | Notes
---------|------------------|----------------|--------------------|--------
CPU      |                  |      Yes       |          Yes       |   
CUDA     | Nvrtx/PTX        |      Yes       |          Yes       |   
Vulkan   | GlSlang/Spir-V   |      Yes       |          Yes       |   
D3D12    | DXC/DXIL         |      Yes       |          Yes       |   1
D3D11    | FXC/DXBC         |      No        |          No        |   2
D3D12    | FXC/DXBC         |      No        |          No        |   2

1) The [sm6.0 docs](https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12) describe only supporting uint64_t, but dxc says int64_t is supported in [HLSL 2016](https://github.com/Microsoft/DirectXShaderCompiler/wiki/Language-Versions). Tests show that this is indeed the case.

2) uint64_t support requires https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12, so DXBC is not a target.

The intrinsics available on `uint64_t` type are `abs`, `min`, `max`, `clamp` and `countbits`.
The intrinsics available on `int64_t` type are `abs`, `min`, `max`, `clamp` and `countbits`.

GLSL
====

GLSL/Spir-v based targets do not support 'generated' intrinsics on matrix types. For example 'sin(mat)' will not work on GLSL/Spir-v.

