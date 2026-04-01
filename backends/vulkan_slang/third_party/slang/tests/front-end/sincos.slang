// sincos.slang
//TEST:SIMPLE:

// Just confirming that calls to the `sincos()` built
// in work in the front-end.

float test( float4x4 m, float4 v, float a )
{
    float4x4 mc, ms;
    sincos(m, mc, ms);

    float4 vc, vs;
    sincos(v, vc, vs);

    float c, s;
    sincos(a, c, s);
	
	return c + s + vc[0] + vs[0] + mc[0][0] + ms[0][0];
}
