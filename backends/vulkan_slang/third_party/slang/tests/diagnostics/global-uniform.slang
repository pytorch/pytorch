// global-uniform.slang
//DIAGNOSTIC_TEST:SIMPLE:-target hlsl
//DIAGNOSTIC_TEST:COMMAND_LINE_SIMPLE:-target hlsl

// An attempt to declare a global variable that actually declares a
// global shader parameter should be diagnosed, unless `uniform` was used.

uniform float a;

const uint4 b = uint4(0,1,2,3);

struct C { float x; int y; };
C c;
