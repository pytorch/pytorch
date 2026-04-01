extern static const bool turnOnFeature;
extern static const float constValue;
extern static const uint numthread = 0;
extern static const int arraySize = -1;

// Main entry-point. Write some value into buffer depending on link
// time constant.
[shader("compute")]
[numthreads(numthread, 1, 1)]
void computeMain(
    uint3 sv_dispatchThreadID: SV_DispatchThreadID,
    uniform RWStructuredBuffer<float> buffer)
{
    int array[arraySize];

    array[sv_dispatchThreadID.x] = sv_dispatchThreadID.x;
    if (turnOnFeature)
    {
        buffer[array[0]] = constValue;
    }
    else
    {
        buffer[0] = -1.0;
    }
}
