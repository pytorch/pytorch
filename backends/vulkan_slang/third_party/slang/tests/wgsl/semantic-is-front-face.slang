//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry main

//WGSL-DAG: @builtin(front_facing)
//WGSL-DAG: @fragment

float4 main(bool isFront : SV_IsFrontFace) : SV_Target 
{
    return float4(1,1,1,1);
}
