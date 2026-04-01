//TEST:SIMPLE(filecheck=CHECK_SPIRV): -target spirv -stage compute -entry main -allow-glsl
//TEST:SIMPLE(filecheck=CHECK_GLSL): -target glsl -stage compute -entry main -allow-glsl

void main()
{
    debugPrintfEXT("test");
    debugPrintfEXT(R"(test1 "%d %d")", 5, 6);

    // CHECK_SPIRV: %[[SET:[0-9]+]] = OpExtInstImport "NonSemantic.DebugPrintf"
    // CHECK_SPIRV: {{.*}} = OpExtInst %{{[a-zA-Z0-9_]+}} %[[SET]] 1 %{{[a-zA-Z0-9_]+}}
    // CHECK_SPIRV: {{.*}} = OpExtInst %{{[a-zA-Z0-9_]+}} %[[SET]] 1 %{{[a-zA-Z0-9_]+}} %int_5 %int_6

    // CHECK_GLSL: debugPrintfEXT("test")
    // CHECK_GLSL: debugPrintfEXT("test1 \"%d %d\"", 5, 6);
}
