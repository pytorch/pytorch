//TEST:INTERPRET(filecheck=CHECK):

#define CHECK(x) if (!(x)) return false

DescriptorAccess getAccess<T : IOpaqueDescriptor>()
{
    return T.descriptorAccess;
}

bool test()
{
    CHECK(getAccess<Texture2D>() == DescriptorAccess.Read);
    CHECK(getAccess<SamplerState>() == DescriptorAccess.Unknown);
    CHECK(getAccess<RasterizerOrderedStructuredBuffer<int>>() == DescriptorAccess.RasterizerOrdered);
    CHECK(getAccess<RWStructuredBuffer<int>>() == DescriptorAccess.ReadWrite);
    CHECK(getAccess<AppendStructuredBuffer<int>>() == DescriptorAccess.ReadWrite);
    CHECK(getAccess<ConsumeStructuredBuffer<int>>() == DescriptorAccess.ReadWrite);
    CHECK(getAccess<StructuredBuffer<int>>() == DescriptorAccess.Read);
    CHECK(getAccess<ConstantBuffer<int>>() == DescriptorAccess.Read);
    CHECK(getAccess<RWByteAddressBuffer>() == DescriptorAccess.ReadWrite);
    CHECK(getAccess<ByteAddressBuffer>() == DescriptorAccess.Read);
    CHECK(getAccess<RaytracingAccelerationStructure>() == DescriptorAccess.Read);
    CHECK(getAccess<WTexture2DArray>() == DescriptorAccess.Write);
    return true;
}

void main()
{
    // CHECK: pass
    if (test())
        printf("pass\n");
    else
        printf("fail\n");
}