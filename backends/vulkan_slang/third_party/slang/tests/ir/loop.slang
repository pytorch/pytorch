//TEST_DISABLED:SIMPLE:-dump-ir -profile cs_5_0 -entry main

// Note: disabling this test for now because
// the actual IR that gets dumped is not very
// stable with code generation changes going on,
// and we already have more significant tests
// that stress the IR functionality.
//
// We should consider removing this test, or
// else work to ensure that "canonical" IR
// output is more consistent.

#define GROUP_THREAD_COUNT 64

StructuredBuffer<float4> input;
RWStructuredBuffer<float4> output;

groupshared float4 s[GROUP_THREAD_COUNT];

[numthreads(GROUP_THREAD_COUNT, 1, 1)]
void main(
    uint dispatchThreadID   : SV_DispatchThreadID,
    uint groupThreadID      : SV_GroupThreadID,
    uint groupID            : SV_GroupIndex )
{
    // the actual algorithm being done here is bogus

    // load shared memory
    s[groupThreadID] = input[dispatchThreadID];

    // do some sum passes
    for(uint stride = 1; stride < GROUP_THREAD_COUNT; stride <<= 1)
    {
        GroupMemoryBarrierWithGroupSync();

        s[groupThreadID] += s[groupThreadID - stride];
    }

    GroupMemoryBarrierWithGroupSync();

    output[dispatchThreadID] = s[0];
}

