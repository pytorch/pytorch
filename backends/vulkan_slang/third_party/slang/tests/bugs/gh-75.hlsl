//TEST:SIMPLE:-profile cs_5_0

// Missing opening `{` sends parser into infinite loop

struct LightCB
{
    float3 vec3Val; // We're using 2 values. [0]: worldDir [1]: intensity
};

StructuredBuffer<LightCB> gLightIn;
AppendStructuredBuffer<LightCB> gLightOut;

[numthreads(1, 1, 1)]
void main()
{
    uint numLights = 0;
    uint stride;
    gLightIn.GetDimensions(numLights, stride);
    
    for (uint i = 0; i < numLights; i++)
    
        gLightOut.Append(gLightIn[i]);
    }
}