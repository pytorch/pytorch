//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage compute -entry main

//WGSL-DAG: @builtin(local_invocation_id)
//WGSL-DAG: @compute

void main(uint3 gtid : SV_GroupThreadID) 
{
    // Empty compute shader
}
