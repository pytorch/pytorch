//TEST:SIMPLE(filecheck=CHECK): -entry MainCs -stage compute -profile glsl_450 -target spirv
//CHECK: EntryPoint

[vk::image_format("rgba8")]
RWTexture2D < unorm float4 > DstColor  ; 

[ numthreads ( 64 , 1 , 1 ) ] 
void MainCs ( uint3 Gid : SV_GroupID , uint GI : SV_GroupIndex , uint3 GTid : SV_GroupThreadID , uint3 DTid : SV_DispatchThreadID ) 
{ 
    DstColor [ 0 ] = float4 ( lerp ( DstColor [ 0 ].rgb, float3( 0, 0, 0 ), 1.0 ) , 1.0f ) ;  
} 
