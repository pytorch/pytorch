// byte-address-buffer-aligned.slang
//TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -d3d12 -profile cs_6_0 -use-dxil -shaderobj -output-using-type
//DISABLED_TEST(compute, vulkan):COMPARE_COMPUTE_EX(filecheck-buffer=CHECK):-slang -compute -vk -shaderobj -output-using-type

//TEST:SIMPLE(filecheck=CHECK1):-target glsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK2):-target hlsl -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK3):-target spirv -entry computeMain -stage compute
//TEST:SIMPLE(filecheck=CHECK3):-target spirv -emit-spirv-directly -entry computeMain -stage compute

// Confirm compilation of `(RW)ByteAddressBuffer` with aligned load / stores to wider data types.

//TEST_INPUT:ubuffer(data=[0 0 0 0], stride=4):out,name=buffer0
[vk::binding(2, 3)] RWByteAddressBuffer buffer0;

[shader("compute")]
[numthreads(1,1,1)]
void computeMain(uint3 threadId : SV_DispatchThreadID)
{
    // CHECK-NOT: warning

    // CHECK1: vec4 {{.*}} = buffer0_{{.*}}.{{.*}} = buffer0_{{.*}}.{{.*}};
    // CHECK1: vec4 {{.*}} = buffer0_{{.*}}.{{.*}} = vec4(buffer0_{{.*}}.{{.*}}, buffer0_{{.*}}.{{.*}}, buffer0_{{.*}}.{{.*}}, buffer0_{{.*}}.{{.*}});
    // CHECK1: vec4 {{.*}} = buffer0_{{.*}}.{{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = _S3[{{.*}}];
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = _S3[{{.*}}];
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = _S3[{{.*}}];
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = _S3[{{.*}}];
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = buffer0_{{.*}}.{{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = {{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = {{.*}};
    // CHECK1: float {{.*}} = buffer0_{{.*}}.{{.*}} = {{.*}};

    // CHECK2: float4 {{.*}} = (buffer0_0).Load<float4 >(int(32));
    // CHECK2: buffer0_0.Store(int(32),{{.*}});
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(8));
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(12));
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(16));
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(20));
    // CHECK2: buffer0_0.Store(int(32),float4({{.*}}, {{.*}}, {{.*}}, {{.*}}));
    // CHECK2: float4 {{.*}} = (buffer0_0).Load<float4 >(int(32));
    // CHECK2: buffer0_0.Store(int(8),{{.*}}[int(0)]);
    // CHECK2: buffer0_0.Store(int(12),{{.*}}[int(1)]);
    // CHECK2: buffer0_0.Store(int(16),{{.*}}[int(2)]);
    // CHECK2: buffer0_0.Store(int(20),{{.*}}[int(3)]);
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(8));
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(12));
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(16));
    // CHECK2: float {{.*}} = (buffer0_0).Load<float >(int(20));
    // CHECK2: buffer0_0.Store(int(8),{{.*}});
    // CHECK2: buffer0_0.Store(int(12),{{.*}});
    // CHECK2: buffer0_0.Store(int(16),{{.*}});
    // CHECK2: buffer0_0.Store(int(20),{{.*}});

    // CHECK3-DAG: %[[v4f:[a-zA-Z0-9_]+]] = OpTypeVector %float 4
    // CHECK3-DAG: %[[SBv4f:[a-zA-Z0-9_]+]] = OpTypePointer StorageBuffer %[[v4f]]
    // CHECK3-DAG: %[[RTv4f:[a-zA-Z0-9_]+]] = OpTypeRuntimeArray %[[v4f]]
    // CHECK3-DAG: %[[RWBv4f:[a-zA-Z0-9_]+]] = OpTypeStruct %[[RTv4f]]
    // CHECK3-DAG: %[[SBRW:[a-zA-Z0-9_]+]] = OpTypePointer StorageBuffer %[[RWBv4f]]
    // CHECK3-DAG: %[[RTf:[a-zA-Z0-9_]+]] = OpTypeRuntimeArray %float
    // CHECK3-DAG: %[[RWBf:[a-zA-Z0-9_]+]] = OpTypeStruct %[[RTf]]
    // CHECK3-DAG: %[[SBRW0:[a-zA-Z0-9_]+]] = OpTypePointer StorageBuffer %[[RWBf]]
    // CHECK3-DAG: %[[SBf:[a-zA-Z0-9_]+]] = OpTypePointer StorageBuffer %float
    // CHECK3-DAG: %[[BUF0:[a-zA-Z0-9_]+]] =  OpVariable %[[SBRW]] StorageBuffer
    // CHECK3-DAG: %[[BUF00:[a-zA-Z0-9_]+]] =  OpVariable %[[SBRW0]] StorageBuffer
    // CHECK3-DAG: %[[V0:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBv4f]] %[[BUF0]] %{{.*}}
    // CHECK3-DAG: %[[V1:[a-zA-Z0-9_]+]] = OpLoad %[[v4f]]  %[[V0]]
    // CHECK3-DAG: %[[V2:[a-zA-Z0-9_]+]] = OpAccessChain  %[[SBv4f]] %[[BUF0]] %{{.*}}
    // CHECK3-DAG: OpStore %[[V2]] %[[V1]]
    // CHECK3-DAG: %[[V3:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V4:[a-zA-Z0-9_]+]] = OpLoad %float %[[V3]]
    // CHECK3-DAG: %[[V5:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V6:[a-zA-Z0-9_]+]] = OpLoad %float %[[V5]]
    // CHECK3-DAG: %[[V7:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V8:[a-zA-Z0-9_]+]] = OpLoad %float %[[V7]]
    // CHECK3-DAG: %[[V9:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V10:[a-zA-Z0-9_]+]] = OpLoad %float %[[V9]]
    // CHECK3-DAG: %[[V11:[a-zA-Z0-9_]+]] = OpCompositeConstruct %[[v4f]] %[[V4]] %[[V6]] %[[V8]] %[[V10]]
    // CHECK3-DAG: %[[V12:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBv4f]] %[[BUF0]]
    // CHECK3-DAG: OpStore %[[V12]] %[[V11]]
    // CHECK3-DAG: %[[V13:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBv4f]] %[[BUF0]]
    // CHECK3-DAG: %[[V14:[a-zA-Z0-9_]+]] = OpLoad %[[v4f]] %[[V13]]
    // CHECK3-DAG: %[[V15:[a-zA-Z0-9_]+]] = OpCompositeExtract %float %[[V14]] 0
    // CHECK3-DAG: %[[V16:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V16]] %[[V15]]
    // CHECK3-DAG: %[[V17:[a-zA-Z0-9_]+]] = OpCompositeExtract %float %[[V14]] 1
    // CHECK3-DAG: %[[V18:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V18]] %[[V17]]
    // CHECK3-DAG: %[[V19:[a-zA-Z0-9_]+]] = OpCompositeExtract %float %[[V14]] 2
    // CHECK3-DAG: %[[V20:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V20]] %[[V19]]
    // CHECK3-DAG: %[[V21:[a-zA-Z0-9_]+]] = OpCompositeExtract %float %[[V14]] 3
    // CHECK3-DAG: %[[V22:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V22]] %[[V21]]
    // CHECK3-DAG: %[[V23:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V24:[a-zA-Z0-9_]+]] = OpLoad %float %[[V23]]
    // CHECK3-DAG: %[[V25:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V26:[a-zA-Z0-9_]+]] = OpLoad %float %[[V25]]
    // CHECK3-DAG: %[[V27:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V28:[a-zA-Z0-9_]+]] = OpLoad %float %[[V27]]
    // CHECK3-DAG: %[[V29:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: %[[V30:[a-zA-Z0-9_]+]] = OpLoad %float %[[V29]]
    // CHECK3-DAG: %[[V31:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V31]] %[[V24]]
    // CHECK3-DAG: %[[V32:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V32]] %[[V26]]
    // CHECK3-DAG: %[[V33:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V33]] %[[V28]]
    // CHECK3-DAG: %[[V34:[a-zA-Z0-9_]+]] = OpAccessChain %[[SBf]] %[[BUF00]]
    // CHECK3-DAG: OpStore %[[V34]] %[[V30]]
    buffer0.StoreAligned(32, buffer0.LoadAligned<float4>(32));
    buffer0.StoreAligned(32, buffer0.LoadAligned<float4>(8));
    buffer0.StoreAligned(8, buffer0.LoadAligned<float4>(32));
    buffer0.StoreAligned(8, buffer0.LoadAligned<float4>(8));
}
