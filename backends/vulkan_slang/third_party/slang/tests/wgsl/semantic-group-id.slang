//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage compute -entry main

//WGSL-DAG: @builtin(workgroup_id)
//WGSL-DAG: @compute

void main(uint3 gid : SV_GroupID) 
{
    // Empty compute shader
}
