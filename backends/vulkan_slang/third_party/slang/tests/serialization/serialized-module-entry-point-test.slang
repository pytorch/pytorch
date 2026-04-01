// serialized-module-entry-point-test.slang

//DISABLED_TEST:COMPILE: -module-name module -target hlsl -profile cs_5_0 -entry computeMain tests/serialization/serialized-module-entry-point.slang -o tests/serialization/serialized-module-entry-point.slang-module
//DISABLED_TEST:COMPILE: -module-name module tests/serialization/serialized-module.slang -o tests/serialization/serialized-module.slang-module -ir-compression none
//DISABLED_TEST:COMPARE_COMPUTE_EX: -xslang -module-name -xslang module -slang -compute -xslang -r -xslang tests/serialization/serialized-module-entry-point.slang-module -xslang -r -xslang tests/serialization/serialized-module.slang-module -no-default-entry-point

//TEST_INPUT:ubuffer(data=[0 0 0 0 ], stride=4):out,name outputBuffer
