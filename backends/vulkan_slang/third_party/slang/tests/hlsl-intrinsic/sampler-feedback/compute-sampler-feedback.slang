//TEST:COMPILE: -entry computeMain -stage compute -target callable tests/hlsl-intrinsic/sampler-feedback/compute-sampler-feedback.slang

// Not available on non PS shader
// dx.op.writeSamplerFeedback 	WriteSamplerFeedback 	
// dx.op.writeSamplerFeedbackBias 	WriteSamplerFeedbackBias

FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMip;
FeedbackTexture2D<SAMPLER_FEEDBACK_MIP_REGION_USED> feedbackMipRegionUsed;
FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIN_MIP> feedbackMinMipArray;
FeedbackTexture2DArray<SAMPLER_FEEDBACK_MIP_REGION_USED> feebackMipRegionUsedArray;

Texture2D<float> tex2D;
Texture2DArray<float> tex2DArray;
SamplerState samp;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    float2 coords2D = float2(1, 2) * dispatchThreadID.x;
    float3 coords2DArray = float3(1, 2, 3) * dispatchThreadID.x;
    
    float clamp = 4;
    float bias = 0.5F;
    float lod = 6;
    float2 ddx = float2(1.0F / 32, 2.0F / 32);
    float2 ddy = float2(3.0F / 32, 4.0F / 32);
    
    uint width, height, elements, mipLevels;
    feedbackMinMip.GetDimensions(width, height);
    feedbackMinMipArray.GetDimensions(width, height, elements);
    
    feedbackMinMip.GetDimensions(0, width, height, mipLevels);
    feedbackMinMipArray.GetDimensions(0, width, height, elements, mipLevels);
    
    
    feedbackMinMip.WriteSamplerFeedbackGrad(tex2D, samp, coords2D, ddx, ddy, clamp);
    
    // Level
    feedbackMinMip.WriteSamplerFeedbackLevel(tex2D, samp, coords2D, lod);
    
    // No Clamp
    feedbackMinMip.WriteSamplerFeedbackGrad(tex2D, samp, coords2D, ddx, ddy);
}
