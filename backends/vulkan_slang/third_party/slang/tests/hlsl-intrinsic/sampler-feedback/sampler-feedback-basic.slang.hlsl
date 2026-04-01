FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMip_0;
FeedbackTexture2D<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackMipRegionUsed_0;
FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMipArray_0;
FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feebackMipRegionUsedArray_0;

Texture2D<float> tex2D_0;
Texture2DArray<float> tex2DArray_0;
SamplerState samp_0;

float4 main() : SV_Target
{
    float2 coords2D = float2(1, 2);
    float3 coords2DArray = float3(1, 2, 3);
    
    float clamp = 4;
    float bias = 0.5F;
    float lod = 6;
    float2 ddx = float2(1.0F / 32, 2.0F / 32);
    float2 ddy = float2(3.0F / 32, 4.0F / 32);
     
    // Clamped
    feedbackMinMip_0.WriteSamplerFeedback(tex2D_0, samp_0, coords2D, clamp);

    feedbackMinMip_0.WriteSamplerFeedbackBias(tex2D_0, samp_0, coords2D, bias, clamp);
    feedbackMinMip_0.WriteSamplerFeedbackGrad(tex2D_0, samp_0, coords2D, ddx, ddy, clamp);
    
    // Level
    feedbackMinMip_0.WriteSamplerFeedbackLevel(tex2D_0, samp_0, coords2D, lod);
    
    // No Clamp
    feedbackMinMip_0.WriteSamplerFeedback(tex2D_0, samp_0, coords2D );
    feedbackMinMip_0.WriteSamplerFeedbackBias(tex2D_0, samp_0, coords2D, bias);
    feedbackMinMip_0.WriteSamplerFeedbackGrad(tex2D_0, samp_0, coords2D, ddx, ddy);

    // Array
    feedbackMinMipArray_0.WriteSamplerFeedback(tex2DArray_0, samp_0, coords2DArray);
    feebackMipRegionUsedArray_0.WriteSamplerFeedback(tex2DArray_0, samp_0, coords2DArray);

    // Using feedbackMipRegionUsed 
    feedbackMipRegionUsed_0.WriteSamplerFeedback(tex2D_0, samp_0, coords2D);
    
    return float4(1, 2, 3, 4);
}