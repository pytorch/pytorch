// attribute.slang

// Tests reflection of user defined attributes.

//TEST:REFLECTION:-stage compute -entry main -target hlsl -no-codegen

[__AttributeUsage(_AttributeTargets.Struct)]
[__AttributeUsage(_AttributeTargets.Function)]
struct MyStructAttribute
{
    int iParam;
    float fParam;
    string sParam;
};

[__AttributeUsage(_AttributeTargets.Var)]
[__AttributeUsage(_AttributeTargets.Param)]
struct DefaultValueAttribute
{
    int iParam;
};

[MyStruct(0, 1.0, "A")]
struct A
{
    float x;
    [DefaultValue(1)]
    float y;
};

[MyStruct(0, 2.0, "\"")]
struct B
{
    float x;
    [DefaultValue(1+1)]
    float z;
};

ParameterBlock<A> param;
ParameterBlock<B> param2;

[DefaultValue(2)] int globalInt;

[__AttributeUsage(_AttributeTargets.Struct)]
[__AttributeUsage(_AttributeTargets.Var)]
[__AttributeUsage(_AttributeTargets.Param)]
struct StructVarParamAttribute
{
    int iParam;
};

[StructVarParam(0)]
struct D
{
    int a;
};

D param3;

[StructVarParam(1)] int globalInt2;


[MyStruct(2, 3.0, "main")]
[MyStruct(-2, -3.0, "")]
[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    [DefaultValue(3)] float a,
    [StructVarParam(2)] float b)
{
}
