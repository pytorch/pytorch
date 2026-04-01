//TEST:SIMPLE(filecheck=CHECK): -target glsl -entry computeMain -stage compute -profile sm_5_0
//TEST:SIMPLE(filecheck=CHECK_IGNORE_CAPS): -target glsl -emit-spirv-directly -entry computeMain -stage compute  -profile sm_5_0 -ignore-capabilities


// CHECK_IGNORE_CAPS-NOT: error 36107

// CHECK: error 36107
// CHECK-SAME: entrypoint 'computeMain' uses features that are not available in 'compute' stage for 'glsl' target.
// CHECK: capabilitySimplification1.slang(21): note: see using of 'WaveMultiPrefixCountBits'
// CHECK-NOT: see using of 'WaveMultiPrefixCountBits'
// CHECK: {{.*}}.meta.slang({{.*}}): note: see definition of 'WaveMultiPrefixCountBits'
// CHECK: {{.*}}.meta.slang({{.*}}): note: see declaration of 'require'

void nestedSafeCall()
{
    AllMemoryBarrier();
}

void nestedBadCall()
{
    WaveMultiPrefixCountBits(true, 0);
}

void nestedCall()
{
    nestedSafeCall();
    nestedBadCall();
}

[numthreads(1,1,1)]
void computeMain()
{
    nestedCall();
}
