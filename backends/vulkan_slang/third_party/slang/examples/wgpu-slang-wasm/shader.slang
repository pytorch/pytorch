struct VertexStageInput
{
    float4 position : POSITION0;
};

struct VertexStageOutput
{
    float4 positionClipSpace : SV_POSITION;
};

struct FragmentStageOutput
{
    float4 color : SV_TARGET;
};

VertexStageOutput vertexMain(VertexStageInput input) : SV_Position
{
	VertexStageOutput output;
    output.positionClipSpace = float4(input.position.xy, 1);
    return output;
}

FragmentStageOutput fragmentMain() : SV_Target
{
    FragmentStageOutput output;
    output.color = float4(0, 1, 0, 1);
    return output;
}
