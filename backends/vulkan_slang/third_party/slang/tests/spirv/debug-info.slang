//TEST:SIMPLE(filecheck=CHECK):-target spirv -entry main -stage compute -g2 -emit-spirv-directly

struct TestType
{
    float memberA;
    float3 memberB;
    RWStructuredBuffer<float> memberC;
    float getValue()
    {
        return memberA;
    }
}
RWStructuredBuffer<float> result;
void main()
{
    TestType t;
    t.memberA = 1.0;
    t.memberB = float3(1, 2, 3);
    t.memberC = result;
    var val = t.getValue();
    result[0] = val + t.memberB.x;
}

// CHECK: OpExtInst %void {{.*}} DebugExpression
// CHECK: DebugTypeMember
// CHECK: DebugTypeComposite
// CHECK: DebugFunctionDefinition
// CHECK: DebugScope
// CHECK: DebugLine
// CHECK: DebugValue
