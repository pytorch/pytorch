
//TEST:SIMPLE(filecheck=CHECK):-target spirv

float ddx_or(float val, float defaultVal)
{
    __stage_switch
    {
        case fragment:
            return ddx(val);
        default:
            return defaultVal;
    }
}

float intermediate(float val)
{
    return ddx_or(val, 1.0);
}

RWStructuredBuffer<float> output;

[numthreads(1,1,1)]
void computeMain()
{
    // CHECK-LABEL: %computeMain = OpFunction 
    // CHECK: OpStore %{{.*}} %float_1
    // CHECK: OpFunctionEnd
    output[0] = intermediate(2.0);
}

[shader("fragment")]
float4 fragmentMain(float vin) : SV_Target
{
    // CHECK-LABEL: %fragmentMain = OpFunction 
    // CHECK: OpDPdx
    // CHECK: OpFunctionEnd
    return intermediate(vin);
}