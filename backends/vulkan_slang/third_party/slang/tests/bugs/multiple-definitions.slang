// multiple-definitions.slang
//TEST:SIMPLE:-entry main -o multiple-definitions.hlsl

__specialized_for_target(hlsl)
int a(int x)
{
    int r = 0;
    for(int i = 0; i < x; ++i) ++r;
    return r;
}

__specialized_for_target(glsl)
int a(int x)
{
    int r = 0;
    for(int i = 0; i < x; ++i) ++r;
    return r;
}

[shader("compute")]
void main()
{}
