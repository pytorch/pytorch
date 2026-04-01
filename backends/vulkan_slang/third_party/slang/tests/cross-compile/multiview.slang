// multiview.slang

// Confirm that the SPV_KHR_multiview and EXT_multiview extensions are compiled
// correctly

//TEST:CROSS_COMPILE:-target spirv-assembly -entry main -stage vertex

void main(uint viewId : SV_ViewID, out float4 p : SV_Position)
{
  p = float4(viewId, 0, 0, 0);
}
