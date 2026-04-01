//TEST(compute, vulkan):COMPARE_COMPUTE(filecheck-buffer=CHECK):-vk -shaderobj

// CHECK:      2
// CHECK-NEXT: 1
// CHECK-NEXT: 4
// CHECK-NEXT: 3
// CHECK-NEXT: 6
// CHECK-NEXT: 5
// CHECK-NEXT: 8
// CHECK-NEXT: 7

struct S
{
    int a;
    int b;
};

//TEST_INPUT:ubuffer(data=[1 2 3 4 5 6 7 8], stride=8):out,name=buf
RWStructuredBuffer<S> buf;

S test(S s)
{
    s.a = s.a ^ s.b;
    s.b = s.a ^ s.b;
    s.a = s.a ^ s.b;
    return s;
}

[numthreads(4, 1, 1)]
// void computeMain(uint i : SV_GroupIndex)
// {
//     buf[i] = test(buf[i]);
// }
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    buf[dispatchThreadID.x] = test(buf[dispatchThreadID.x]);
}
