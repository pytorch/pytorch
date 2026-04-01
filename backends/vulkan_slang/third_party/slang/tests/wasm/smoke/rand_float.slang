// Hybrid Tausworthe PRNG
//
// Code adapted from: https://indico.cern.ch/event/93877/papers/2118070/files/4416-acat3.pdf (See document for license)
//

uniform float seed;
RWStructuredBuffer<float> outputBuffer;

uint seedPerThread(uint idx)
{
    return ((uint)idx + (uint)(seed * 1000000)) * 1099087573UL;
}

uint tauStep(uint z, uint s1, uint s2, uint s3, uint M)
{
    uint b = (((z << s1) ^ z) >> s2);
    return (((z & M) << s3) ^ b);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint2 dispatchThreadId : SV_DispatchThreadID)
{
    uint idx = dispatchThreadId.x;
    uint val = ((uint)idx) * 1099087573UL + ((uint)seed) * 12003927;

    uint z = tauStep(val, 13, 19, 12, 4294967294);
    z = tauStep(z, 2, 25, 4, 4294967288);
    z = tauStep(z, 3, 11, 17, 4294967280);

    uint z1, z2, z3, z4;
    uint r0, r1, r2, r3;

    // STEP 1
    uint _seed = seedPerThread(idx);
    z1 = tauStep(_seed, 13, 19, 12, 429496729UL);
    z2 = tauStep(_seed, 2, 25, 4, 4294967288UL);
    z3 = tauStep(_seed, 3, 11, 17, 429496280UL);
    z4 = (1664525 * _seed + 1013904223UL);
    r0 = (z1 ^ z2 ^ z3 ^ z4);
    // STEP 2
    z1 = tauStep(r0, 13, 19, 12, 429496729UL);
    z2 = tauStep(r0, 2, 25, 4, 4294967288UL);
    z3 = tauStep(r0, 3, 11, 17, 429496280UL);
    z4 = (1664525 * r0 + 1013904223UL);
    r1 = (z1 ^ z2 ^ z3 ^ z4);
    // STEP 3
    z1 = tauStep(r1, 13, 19, 12, 429496729UL);
    z2 = tauStep(r1, 2, 25, 4, 4294967288UL);
    z3 = tauStep(r1, 3, 11, 17, 429496280UL);
    z4 = (1664525 * r1 + 1013904223UL);
    r2 = (z1 ^ z2 ^ z3 ^ z4);
    // STEP 4
    z1 = tauStep(r2, 13, 19, 12, 429496729UL);
    z2 = tauStep(r2, 2, 25, 4, 4294967288UL);
    z3 = tauStep(r2, 3, 11, 17, 429496280UL);
    z4 = (1664525 * r2 + 1013904223UL);
    r3 = (z1 ^ z2 ^ z3 ^ z4);

    float u4 = r3 * 2.3283064365387e-10;

    outputBuffer[idx] = u4;
}