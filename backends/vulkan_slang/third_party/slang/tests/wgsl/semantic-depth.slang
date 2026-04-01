//TEST:SIMPLE(filecheck=WGSL): -target wgsl -stage fragment -entry main

//WGSL-DAG: @builtin(frag_depth)
//WGSL-DAG: @fragment

struct Out 
{
    float depth : SV_Depth;
};

Out main() 
{
    Out output;
    output.depth = 1.0;
    return output;
}
