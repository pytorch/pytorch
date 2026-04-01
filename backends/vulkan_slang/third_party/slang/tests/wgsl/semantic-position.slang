//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage vertex -entry main

//WGSL-DAG: @builtin(position)
//WGSL-DAG: @vertex

struct S 
{
    float4 position : SV_Position;
};

S main() 
{
    S output;
    output.position = float4(0,0,0,1);
    return output;
}
