layout(std430) buffer;
layout(FORMAT, binding=0) writeonly uniform PRECISION image3D uOutput;
layout(location=1) uniform PRECISION sampler3D uInput;

layout(location=2) uniform ivec3 uInputSize;
layout(location=3) uniform ivec3 uOutputSize;

layout(location=4) uniform float uScaleX;
layout(location=5) uniform float uScaleY;

layout(local_size_x = WORKGROUP_X, local_size_y = WORKGROUP_Y, local_size_z = WORKGROUP_Z) in;

void main()
{
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  if(pos.x < uOutputSize.x && pos.y < uOutputSize.y)
  {
    float srcX = float(pos.x) * uScaleX;
    int x1 = int(floor(srcX));
    int x11 = clamp(x1, 0, uInputSize.x - 1);
    float srcY = float(pos.y) * uScaleY;
    int y1 = int(floor(srcY));
    int y11 = clamp(y1, 0, uInputSize.y - 1);
    vec4 outValue = texelFetch(uInput, ivec3(x11, y11, pos.z), 0);
    imageStore(uOutput, pos, outValue);
  }
}
