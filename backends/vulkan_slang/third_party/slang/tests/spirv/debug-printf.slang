//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main -emit-spirv-directly
//TEST:SIMPLE(filecheck=CHECK): -target spirv -stage compute -entry main -emit-spirv-via-glsl

void main()
{
    printf("test");
    printf(R"(test1 "%d %d")", 5, 6);
    // CHECK: %[[SET:[0-9]+]] = OpExtInstImport "NonSemantic.DebugPrintf"
    // CHECK: {{.*}} = OpExtInst %{{[a-zA-Z0-9_]+}} %[[SET]] 1 %{{[a-zA-Z0-9_]+}}
    // CHECK: {{.*}} = OpExtInst %{{[a-zA-Z0-9_]+}} %[[SET]] 1 %{{[a-zA-Z0-9_]+}} %int_5 %int_6
}
