//TEST_IGNORE_FILE:

// Imported code used by `reflect-imported-code.hlsl`

public float4 use(float4 val) { return val; };
public float4 use(Texture2D t, SamplerState s) { return t.Sample(s, 0.0); }

public Texture2D t_i;
public SamplerState 	s_i;

public cbuffer C_i
{
    public float c_i;
}
