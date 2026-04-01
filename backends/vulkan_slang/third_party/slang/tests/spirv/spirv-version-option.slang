// spirv-version-option.slang

// Test for command-line options to control SPIR-V version used.

//TEST:CROSS_COMPILE:-target spirv -profile glsl_450 -entry main -stage compute
//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+spirv_1_1 -entry main -stage compute
//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+spirv_1_2 -entry main -stage compute
//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+spirv_1_3 -entry main -stage compute
//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+spirv_1_4 -entry main -stage compute
//TEST:CROSS_COMPILE:-target spirv -profile glsl_450+spirv_1_5 -entry main -stage compute

[shader("compute")]
void main()
{}