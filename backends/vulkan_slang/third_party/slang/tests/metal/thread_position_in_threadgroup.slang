//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target metal -D GROUPID
//TEST:SIMPLE(filecheck=CHECK): -entry computeMain -stage compute -target metal

//CHECK: computeMain

// ensure we compute the SV_GroupIndex from SV_GroupThreadID and `numthreads`
// CHECK: thread_position_in_threadgroup
// CHECK-DAG: *{{.*}}2

RWBuffer<uint> dst;

void indirection(uint groupIndex)
{
    dst[groupIndex] = groupIndex;
}

#define THREAD_COUNT 2
[numthreads(THREAD_COUNT, 1, 1)]
#ifdef GROUPID
void computeMain(uint GI : SV_GroupIndex, uint GTID : SV_GroupThreadID)
#else
void computeMain(uint GI : SV_GroupIndex)
#endif
{
    dst[GI + THREAD_COUNT] = GI;
    indirection(GI);
} 