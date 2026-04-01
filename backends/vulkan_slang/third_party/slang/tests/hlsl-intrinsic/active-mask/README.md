Active Mask Tests
=================

The tests in this directory are designed to ensure that the "active mask" used by HLSL wave-level operations matches what is expected, even on targets where the active mask must be synthesized.

Note that the exact active mask that should be used on wave operations isn't precisely defined in documentation for HLSL. The nearest thing to a public statement of the intended behavior is this statement on the [wiki for dxc](https://github.com/Microsoft/DirectXShaderCompiler/wiki/Wave-Intrinsics) (emphasis ours):

> These intrinsics are dependent on active lanes and therefore flow control. In the model of this document, implementations must enforce that the number of active lanes exactly corresponds to the *programmer’s view of flow control*. In a future version, there may be a compiler flag to relax this requirement as a default, but also enable applications to be explicit about the exact set of lanes to be used in a particular wave operation (see section Wave Handles in the Future Features section below).

The requirement is then to compute an explicit mask that matches the "programmers view" of control flow, which is arguably something up to interpretation.

The GLSL "subgroup" operations are slightly more precise in the language they use, but ultimately leaves the expected value of the active mask under-specified in many cases.

The goal of these tests is to establish some empirical results for what the active mask is expected/required to be in various cases. We will do our best to match the observed behavior of APIs where the implicit "active mask" is an existing feature, but we also reserve the right to take a stand and define what the behavior *ought* to be based on the necessarily more precise definitions that we use in the Slang implementation.
