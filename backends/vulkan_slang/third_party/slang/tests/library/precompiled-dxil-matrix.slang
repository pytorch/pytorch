// precompiled-dxil-matrix.slang

// This test imports a precompiled module that exports a matrix type, which is known to
// cause https://github.com/shader-slang/slang/issues/4880 without driver mitigation.

//TEST(windows):COMPILE: tests/library/module-library-matrix.slang -o tests/library/module-library-matrix.slang-module -target dxil -embed-downstream-ir -profile lib_6_6 -incomplete-library
//TEST(windows):COMPILE: tests/library/precompiled-dxil-matrix.slang -stage anyhit -entry shadow -target dxil -o precompiled-dxil-matrix.dxil

import "module-library-matrix";

struct ShadowHitInfo
{
    bool isHit;
    uint seed;
};

struct Attributes
{
    float2 bary;
};

[shader("anyhit")]
void shadow(inout ShadowHitInfo payload, Attributes attrib)
{
    IgnoreHit();
}
