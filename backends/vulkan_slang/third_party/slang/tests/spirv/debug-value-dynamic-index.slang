//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage compute -g2 -emit-spirv-directly

struct TestType
{
    float memberA;
    float3 memberB;
    float arrayVal[10];
    RWStructuredBuffer<float> memberC;
    float getValue()
    {
        return memberA;
    }
}
RWStructuredBuffer<float> result;
void main(int id : SV_DispatchThreadID)
{
    TestType t;
    t.memberA = 1.0;
    t.arrayVal[id] = 2;
    result[0] = t.arrayVal[id];
}

// CHECK: OpExtInst %void {{.*}} DebugExpression
// CHECK: DebugTypeMember
// CHECK: DebugTypeComposite
// CHECK: DebugFunctionDefinition
// CHECK: DebugScope
// CHECK: DebugLine
// CHECK: DebugValue
