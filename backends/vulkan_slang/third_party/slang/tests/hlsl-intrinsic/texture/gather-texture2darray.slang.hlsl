// gather-texture2darray.slang.hlsl

//TEST_IGNORE_FILE:

Texture2DArray<uint> t_0;
SamplerState s_0;
RWBuffer<uint4> b_0;

// Attribute not understood by fxc
//[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 tid : SV_DISPATCHTHREADID)
{
    b_0[tid.x] = t_0.Gather(s_0, tid);
}
