// precompiled-dxil.slang

// A test that uses slang-modules with embedded precompiled DXIL.
// The test compiles both library slang (export-library.slang) to a slang-module using -embed-downstream-ir.
// The result is linked together with this module (precompiled-dxil.slang) in a second slangc invocation.
// Internally, slang does not use the IR in the modules to link, but rather their embedded DXIL.
// The test passes if there is no errror thrown.
// TODO: Check if final linkage used only the precompiled dxil.

//TEST(windows):COMPILE: tests/library/export-library.slang -o tests/library/export-library.slang-module -target dxil -embed-downstream-ir -profile lib_6_6 -incomplete-library
//TEST(windows):COMPILE: tests/library/precompiled-dxil.slang tests/library/export-library.slang-module -target dxil -entry computeMain -profile cs_6_6 -o tests/library/linked.dxil

extern int foo(int a);

RWStructuredBuffer<int> outputBuffer;

[shader("compute")]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = (int)dispatchThreadID.x;

    outputBuffer[index] = foo(index);
}
