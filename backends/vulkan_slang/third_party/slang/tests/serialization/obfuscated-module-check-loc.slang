//TEST:COMPILE: tests/serialization/obfuscated-loc-module.slang -o tests/serialization/obfuscated-loc-module.zip -g -obfuscate 
//TEST:SIMPLE(filecheck=CHECK):-target hlsl -stage compute -entry computeMain -obfuscate -r tests/serialization/obfuscated-loc-module.zip 
//TEST:COMPILE: tests/serialization/obfuscated-loc-module.slang -o tests/serialization/obfuscated-loc-module.slang-module -g -obfuscate 
//TEST:SIMPLE(filecheck=CHECK):-target hlsl -stage compute -entry computeMain -obfuscate -r tests/serialization/obfuscated-loc-module.slang-module
//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

// CHECK: {{.*}}: error 40020: loop does not terminate within the limited number of iterations, unrolling is aborted.

// This test checks obfuscated source map loc tracking through a round trip, and producing a location correctly from slang-module that has a source map

// We *don't* import because if we do we'll get a fresh compilation from source... we want to make sure it's using the -r module
//import obfuscated_loc_module;
extern int silly(int v);
extern int billy(int v);

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int x = int(dispatchThreadID.x);
    
    x = billy(x);
    
    // Will produce an error, because silly has an error.
    int v = silly(x);
    
    outputBuffer[x] = v;
}

