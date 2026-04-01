//TEST:REFLECTION: -entry main -stage fragment -profile sm_6_5 -target hlsl -no-codegen
//DISABLE_TEST:CROSS_COMPILE: -entry main -stage fragment -profile sm_6_5 -target dxil-assembly

FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMip;
FeedbackTexture2D<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackMipRegionUsed;
FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMipArray;
FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feebackMipRegionUsedArray;

Texture2D<float> tex2D;
Texture2DArray<float> tex2DArray;
SamplerState samp;

float4 main() : SV_Target
{
    float2 coords2D = float2(1, 2);
    float3 coords2DArray = float3(1, 2, 3);
    
    float clamp = 4;
    float bias = 0.5F;
    float lod = 6;
    float2 ddx = float2(1.0F / 32, 2.0F / 32);
    float2 ddy = float2(3.0F / 32, 4.0F / 32);
    
    uint width, height, elements;
    feedbackMinMip.GetDimensions(width, height);
     
    feedbackMinMipArray.GetDimensions(width, height, elements);
     
    // Clamped
    feedbackMinMip.WriteSamplerFeedback(tex2D, samp, coords2D, clamp);

    feedbackMinMip.WriteSamplerFeedbackBias(tex2D, samp, coords2D, bias, clamp);
    feedbackMinMip.WriteSamplerFeedbackGrad(tex2D, samp, coords2D, ddx, ddy, clamp);
    
    // Level
    feedbackMinMip.WriteSamplerFeedbackLevel(tex2D, samp, coords2D, lod);
    
    // No Clamp
    feedbackMinMip.WriteSamplerFeedback(tex2D, samp, coords2D );
    feedbackMinMip.WriteSamplerFeedbackBias(tex2D, samp, coords2D, bias);
    feedbackMinMip.WriteSamplerFeedbackGrad(tex2D, samp, coords2D, ddx, ddy);

    // Array
    feedbackMinMipArray.WriteSamplerFeedback(tex2DArray, samp, coords2DArray);
    feebackMipRegionUsedArray.WriteSamplerFeedback(tex2DArray, samp, coords2DArray);

    // Using feedbackMipRegionUsed 
    feedbackMipRegionUsed.WriteSamplerFeedback(tex2D, samp, coords2D);
    
    return float4(1, 2, 3, 4);
}
