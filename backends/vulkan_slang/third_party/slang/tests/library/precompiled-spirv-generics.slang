// precompiled-spirv-generics.slang

// A test that uses slang-modules with embedded precompiled SPIRV and a library containing generics.
// The test compiles a library slang (export-library-generics.slang) with -embed-downstream-ir then links the
// library to entrypoint slang (this file).
// The test passes if there is no errror thrown.
// TODO: Check if final linkage used only the precompiled spirv.

//TEST:COMPILE: tests/library/export-library-generics.slang -o tests/library/export-library-generics.slang-module -target spirv -embed-downstream-ir -profile lib_6_6 -incomplete-library
//TEST:COMPILE: tests/library/precompiled-spirv-generics.slang -target spirv -stage anyhit -entry anyhit -o tests/library/linked.spirv

import "export-library-generics";

struct Payload
{
    int val;
}

struct Attributes
{
    float2 bary;
}

[shader("anyhit")]
void anyhit(inout Payload payload, Attributes attrib)
{
    payload.val = normalFunc(floor(x * y), x) + normalFuncUsesGeneric(y);
}
