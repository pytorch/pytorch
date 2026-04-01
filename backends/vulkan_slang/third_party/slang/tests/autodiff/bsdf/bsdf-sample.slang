//TEST(compute, vulkan):COMPARE_COMPUTE_EX:-vk -compute -shaderobj -output-using-type
//TEST(compute):COMPARE_COMPUTE_EX:-slang -compute -shaderobj -output-using-type

//TEST_INPUT:ubuffer(data=[0 0 0 0 0 0 0 0 0 0], stride=4):out,name=outputBuffer
module "bsdf-sample";
__include bsdf_auto_rev;

RWStructuredBuffer<float> outputBuffer;

[numthreads(1, 1, 1)]
void computeMain(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    ShadingData sd;
    sd.N = float3(0, 0, 1);
    sd.T = float3(1, 0, 0);
    sd.B = float3(0, 1, 0);
    sd.V = normalize(float3(0.3, 0.5, 0.8));

    {
        DifferentialPair<Auto_Bwd_BSDFParameters> dp_params = DifferentialPair<Auto_Bwd_BSDFParameters>(
            { float3(0.9, 0.6, 0.4), 0.1 },
            { float3(0, 0, 0), 0 });

        Auto_Bwd_ScatterSample.Differential dOut = { float3(0, 0, 0), 0, float3(1, 0, 0) };
        __bwd_diff(bsdfGGXSample)(sd, dp_params, dOut);

        outputBuffer[0] = dp_params.d.albedo[0];
        outputBuffer[1] = dp_params.d.albedo[1];
        outputBuffer[2] = dp_params.d.albedo[2];
        outputBuffer[3] = dp_params.d.roughness;
    }

    {
        DifferentialPair<Auto_Bwd_BSDFParameters> dp_params = DifferentialPair<Auto_Bwd_BSDFParameters>(
            { float3(0.9, 0.6, 0.4), 0.1 },
            { float3(0, 0, 0), 1.0 });
        DifferentialPair<Auto_Bwd_ScatterSample> dp_result;
        __fwd_diff(bsdfGGXSample)(sd, dp_params, dp_result);

        outputBuffer[4] = dp_result.p.weight[0];
        outputBuffer[5] = dp_result.p.weight[1];
        outputBuffer[6] = dp_result.p.weight[2];
        outputBuffer[7] = dp_result.d.weight[0];
        outputBuffer[8] = dp_result.d.weight[1];
        outputBuffer[9] = dp_result.d.weight[2];
    }
}
