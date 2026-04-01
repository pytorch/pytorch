//TEST_IGNORE_FILE:
RWTexture2D<half > halfTexture_0 : register(u1);
RWTexture2D<vector<half,2> > halfTexture2_0 : register(u2);
RWTexture2D<vector<half,4> > halfTexture4_0 : register(u3);

RWStructuredBuffer<int > outputBuffer_0 : register(u0);

[shader("compute")][numthreads(4, 4, 1)]
void computeMain(uint3 dispatchThreadID_0 : SV_DISPATCHTHREADID)
{
    int2 pos_0 = int2(dispatchThreadID_0.xy);
    float _S1 = 1.00000000000000000000 / 3.00000000000000000000;
    int _S2 = pos_0.y;
    int _S3 = pos_0.x;
    int2 pos2_0 = int2(int(3) - _S2, int(3) - _S3);

    half h_0 = halfTexture_0[uint2(pos2_0)];
    vector<half, 2> h2_0 = halfTexture2_0[uint2(pos2_0)];
    vector<half, 4> h4_0 = halfTexture4_0[uint2(pos2_0)];

    halfTexture_0[uint2(pos_0)] = h2_0.x + h2_0.y;
    halfTexture2_0[uint2(pos_0)] = h4_0.xy;
    halfTexture4_0[uint2(pos_0)] = vector<half, 4>(h2_0, h_0, h_0);

    int index_0 = _S3 + _S2 * int(4);
    outputBuffer_0[uint(index_0)] = index_0;
    return;
}
