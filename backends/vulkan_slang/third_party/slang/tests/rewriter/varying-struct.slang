//TEST_IGNORE_FILE:

struct VS_IN
{
	float4 x : X;
	float4 y : Y;	
};

struct VS_OUT
{
	float4 color : COLOR;
	float4 posH : SV_Position;
};

VS_OUT doIt(VS_IN i)
{
	VS_OUT o;
	o.color = i.x;
	o.posH = i.y;
	return o;
}
