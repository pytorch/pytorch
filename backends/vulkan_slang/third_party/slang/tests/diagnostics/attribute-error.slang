// attribute.slang

// Tests reflection of user defined attributes.

//TEST:SIMPLE(filecheck=REFLECTION):-stage compute -entry main -target hlsl

[__AttributeUsage(_AttributeTargets.Struct)]
struct MyStructAttribute
{
    int iParam;
    float fParam;
};
[__AttributeUsage(_AttributeTargets.Var)]
struct DefaultValueAttribute
{
    int iParam;
};

//REFLECTION:([[# @LINE+1]]): error 30019: expected an expression of type 'float', got 'String'
[MyStruct(0, "stringVal")] // attribute arg type mismatch
struct A
{
    //REFLECTION:([[# @LINE+1]]): error 31002: attribute 'MyStruct' is not valid here
    [MyStruct(0, 10.0)] // attribute does not apply to this construct
    float x;
    [DefaultValue(2.0)] // attribute arg type mismatch
    float y;
};

ParameterBlock<A> param;

[numthreads(1, 1, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID)
{
}
