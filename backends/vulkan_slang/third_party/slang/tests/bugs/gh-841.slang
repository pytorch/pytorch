// gh-841.slang

//TEST:CROSS_COMPILE(filecheck=SPV): -profile ps_5_0 -entry main -target spirv-assembly
//TEST:CROSS_COMPILE(filecheck=GLSL): -profile ps_5_0 -entry main -target glsl

// GitHub issue #841: failing to emit `flat` modifier in output GLSL when required

struct RasterVertex
{
  // location 0
	float4 c : COLOR;

  // Make sure that the input value in location 1 is decorated as Flat
  // SPV-DAG: OpDecorate [[VAL:%[_A-Za-z0-9]+]] Location 1
  // SPV-DAG: [[VAL]] = OpVariable {{.*}} Input
  // SPV-DAG: OpDecorate [[VAL]] Flat
  //
  // Likewise for GLSL
  // GLSL:      flat layout(location = 1)
  // GLSL-NEXT: in uint {{[[:alnum:]_]+}};
  //
  // location 1
	uint u : FLAGS;
};

float4 main(RasterVertex v) : SV_Target
{
	float4 result = v.c;
	if((v.u & 1) != 0)
	    result += 1.0;
   return result;
}
