//TEST(compute, vulkan):SIMPLE(filecheck=SPV): -stage compute -entry computeMain -target spirv -emit-spirv-directly -g2
//DISABLE_TEST(compute, vulkan):SIMPLE(filecheck=SPV): -stage compute -entry computeMain -target spirv -emit-spirv-directly -g2 -zero-initialize

// This test is to check if DebugTypePointer is emitted when "-g2" option is used

//TEST_INPUT:ubuffer(data=[0], stride=4):out,name=outputBuffer
RWStructuredBuffer<float> outputBuffer;

//SPV:OpExtension "SPV_KHR_relaxed_extended_instruction"
//SPV: [[STRING_float:%[1-9][0-9]*]] = OpString "float"
//SPV: [[STRING_pValue:%[1-9][0-9]*]] = OpString "pValue"
//SPV: [[STRING_LinkedNode:%[1-9][0-9]*]] = OpString "LinkedNode{{.*}}"
//SPV: [[STRING_pNext:%[1-9][0-9]*]] = OpString "pNext"

struct LinkedNode
{
    float value;

    //SPV: [[TYPE_float:%[1-9][0-9]*]] = OpExtInst %void %{{[0-9]*}} DebugTypeBasic [[STRING_float]]
    //SPV: [[TYPE_float_ptr:%[1-9][0-9]*]] = OpExtInst %void %{{[0-9]*}} DebugTypePointer [[TYPE_float]]
    //SPV: DebugTypeMember [[STRING_pValue]] [[TYPE_float_ptr]]
    float *pValue;

    // "MEMBER_pNext" hasn't been declared yet, but OpExtInstWithForwardRefs requires at least one of these "forward declared ID".
    //SPV: [[FORWARD_LinkedNode:%[1-9][0-9]*]] = OpExtInstWithForwardRefsKHR %void %{{[0-9]*}} DebugTypeComposite [[STRING_LinkedNode]] {{.*}} [[MEMBER_pNext:%[1-9][0-9]*]]
    //SPV: [[TYPE_pLinkedNode:%[1-9][0-9]*]] = OpExtInst %void %{{[0-9]*}} DebugTypePointer [[FORWARD_LinkedNode]]
    //SPV: [[MEMBER_pNext]] = {{.*}} DebugTypeMember [[STRING_pNext]] [[TYPE_pLinkedNode]]
    LinkedNode *pNext;
};

//SPV:  OpExtInst %void %{{[0-9]*}} DebugTypeComposite [[STRING_LinkedNode]] {{.*}} [[MEMBER_pNext]]

float test(LinkedNode *pNode)
{
    LinkedNode *pNodeNext = pNode->pNext;
    return *(pNode->pValue) + pNodeNext->value;
}

cbuffer Constants
{
    LinkedNode *head;
};

[numthreads(1,1,1)]
void computeMain()
{
    LinkedNode node; // required to emit DebugTypeXXX
    outputBuffer[0] = test(head);
}

