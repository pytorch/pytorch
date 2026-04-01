//TEST:SIMPLE(filecheck=CHK): -target spirv-asm -stage compute -entry main -skip-spirv-validation
struct RecordData
{
    int myData;
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadId : SV_GroupThreadID)
{
    spirv_asm
    {
        OpExecutionMode $main ShaderIndexAMDX $(0);
        OpExecutionMode $main StaticNumWorkgroupsAMDX $(1) $(1) $(1);
    };

    DispatchNodeInputRecord<RecordData> inputData;

    let recordData = inputData.Get();
    int myData = recordData.myData;
}

//CHK: ; Types, variables and constants
//CHK: [[MemberType:%[a-zA-Z_0-9]+]] = OpTypeInt 32 1
//CHK: [[StructType:%[a-zA-Z_0-9]+]] = OpTypeStruct [[MemberType]]
//CHK: [[PayloadType:%[a-zA-Z_0-9]+]] = OpTypeNodePayloadArrayAMDX [[StructType]]
//CHK: [[PtrType:%[a-zA-Z_0-9]+]] = OpTypePointer NodePayloadAMDX [[PayloadType]]

//CHK: ; Function
//CHK: [[VarName:%[a-zA-Z_0-9]+]] = OpVariable [[PtrType]] NodePayloadAMDX
//CHK: = OpAccessChain [[PtrType]] [[VarName]]

