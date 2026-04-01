//TEST:SIMPLE(filecheck=CHECK): -target metal -fvk-use-entrypoint-name

struct FirstStruct {
    uint a;
}

struct SecondStruct {
    uint a;
}

struct ThirdStruct {
    uint a;
}

struct FourthStruct {
    uint a;
}

StructuredBuffer<uint> globalScopeBuffer;

RWStructuredBuffer<uint> outBuffer;

//
// Checks for the following:
//  - Output entry points will only contain parameters that they originally have from the Slang source. 
//  - Binding offset calculation for global params originating from entry point parameters are done per-whole file.
//  - Binding offset for global params in global scope are the same for each generated entry point.
//

// CHECK: main1({{.*}}globalScopeBuffer{{.*}}buffer(0)]], FirstStruct{{.*}}buffer(2)]], SecondStruct{{.*}}buffer(3)]], float{{.*}}buffer(4)]])
[shader("compute")]
[numthreads(5, 1, 1)]
void main1(
    uint3 dispatchThreadID: SV_DispatchThreadID,
    RWStructuredBuffer<FirstStruct> custom,
    RWStructuredBuffer<SecondStruct> other,
    ConstantBuffer<float> factor
) {
    uint index = dispatchThreadID.x;
    outBuffer[index] = globalScopeBuffer[index] * custom[index].a * other[index].a * uint(factor);
}


// CHECK-NOT: FirstStruct
// CHECK-NOT: SecondStruct
// CHECK: main2({{.*}}globalScopeBuffer{{.*}}buffer(0)]], ThirdStruct{{.*}}buffer(5)]], FourthStruct{{.*}}buffer(6)]])
[shader("compute")]
[numthreads(5, 1, 1)]
void main2(
    uint3 dispatchThreadID: SV_DispatchThreadID,
    RWStructuredBuffer<ThirdStruct> custom,
    RWStructuredBuffer<FourthStruct> other,
) {
    uint index = dispatchThreadID.x;
    outBuffer[index] = globalScopeBuffer[index] * custom[index].a * other[index].a;
}

