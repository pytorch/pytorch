//TEST:SIMPLE:

// undefined identifier used in `[instance(...)]` attribute

struct V { float4 u : U; };

[instance(COUNT)]
void myGS(triangle V input[3], TriangleStream<V> output)
{
	
}
