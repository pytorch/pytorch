// obfuscated-serialized-module-test.slang

// A test to try out the basics of module
// serialization, obfuscation and source maps.

//TEST:COMPILE: tests/serialization/obfuscated-serialized-module.slang -o tests/serialization/obfuscated-serialized-module.slang-module -g -obfuscate
//TEST:COMPARE_COMPUTE_EX:-slang -compute -Xslang... -r tests/serialization/obfuscated-serialized-module.slang-module -obfuscate -X. -shaderobj

import obfuscated_serialized_module_shared;

extern int foo(Thing thing);

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer
RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Thing thing;

    int index = (int)dispatchThreadID.x;
        
    thing.a = index;
    thing.b = -index;

    outputBuffer[index] = foo(thing);
}
