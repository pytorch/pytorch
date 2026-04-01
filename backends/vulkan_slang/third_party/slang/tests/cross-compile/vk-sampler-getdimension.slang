// vk-sampler-getdimension.slang

//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage fragment -emit-spirv-directly

//  Input to the fragment shader.
struct PSin
{
  float3 position : POSITION;
};

// Output of the fragment shader
struct PSout
{
  float4 color : SV_Target;
};

[[vk::binding(1)]] Sampler3D g_Volume;

// Fragment Shader
[shader("pixel")]
PSout main(PSin stage, bool isFrontFacing : SV_IsFrontFace)
{
  // CHECK: %image{{.*}} = OpImage
  // CHECK: OpImageQuerySizeLod %{{.*}} %image{{.*}}
  PSout output;
  // Find normal at position
  uint3 dim;
  uint levels;
  g_Volume.GetDimensions(0, dim.x, dim.y, dim.z, levels);
  output.color = g_Volume.Sample(float3(dim));
  
  return output;
}
