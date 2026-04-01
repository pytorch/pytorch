//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage vertex -entry main

//WGSL-DAG: @builtin(instance_index)
//WGSL-DAG: @vertex

float4 main(uint instanceID : SV_InstanceID) : SV_Position 
{
    return float4(1,1,1,1);
}
