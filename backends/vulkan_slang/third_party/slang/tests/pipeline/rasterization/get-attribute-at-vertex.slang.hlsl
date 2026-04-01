// get-attribute-at-vertex.slang.hlsl

//TEST_IGNORE_FILE:

[shader("pixel")]
void main(
    nointerpolation vector<float,4> color_0 : COLOR,
    vector<float,3> bary_0 : SV_BARYCENTRICS,
    out vector<float,4> result_0 : SV_TARGET)
{
    result_0 = bary_0.x * GetAttributeAtVertex(color_0, 0U)
             + bary_0.y * GetAttributeAtVertex(color_0, 1U)
             + bary_0.z * GetAttributeAtVertex(color_0, 2U);
}