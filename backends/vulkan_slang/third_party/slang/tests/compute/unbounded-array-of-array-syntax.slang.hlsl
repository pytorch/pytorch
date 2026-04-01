//TEST_IGNORE_FILE:

#pragma pack_matrix(column_major)

RWStructuredBuffer<int >  g_aoa_0[] : register(u0, space1);

RWStructuredBuffer<int > outputBuffer_0 : register(u0);

[numthreads(8, 1, 1)]
void computeMain(vector<uint,3> dispatchThreadID_0 : SV_DISPATCHTHREADID)
{
    int innerIndex_0;

    int index_0 = (int) dispatchThreadID_0.x;

    int innerIndex_1 = index_0 & 3;

    RWStructuredBuffer<int > buffer_0 = g_aoa_0[NonUniformResourceIndex(index_0 >> 2)];

    uint _S1;
    uint _S2;

    buffer_0.GetDimensions(_S1, _S2);

    uint bufferCount_0 = _S1;

    if(innerIndex_1 >= (int)bufferCount_0)
    {
        innerIndex_0 = (int) (bufferCount_0 - (uint) 1);
    }
    else
    {
        innerIndex_0 = innerIndex_1;
    }

    uint _S3 = (uint) innerIndex_0;
    outputBuffer_0[(uint) index_0] = buffer_0[_S3];
    return;
}

