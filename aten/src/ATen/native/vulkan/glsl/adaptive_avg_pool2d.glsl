#version 450 core
layout(std430) buffer;
layout(std430) uniform;
layout(set = 0, rgba16f, binding = 0) writeonly highp uniform image3D uOutput;
layout(set = 0, binding = 1) uniform highp sampler3D uInput;
layout(set = 0, binding = 2) uniform constBlock {
  int IW;
  int IH;
  int OW;
  int OH;
}
uConstBlock;

layout(local_size_x_id = 1, local_size_y_id = 2, local_size_z_id = 3) in;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  int ow = uConstBlock.OW;
  int oh = uConstBlock.OH;
  if (pos.x < ow && pos.y < oh) {
    int iw = uConstBlock.IW;
    int ih = uConstBlock.IH;

    int sx = int(floor(float(pos.x * iw) / ow));
    int sy = int(floor(float(pos.y * ih) / oh));
    int ex = int(ceil(float((pos.x + 1) * iw) / ow));
    int ey = int(ceil(float((pos.y + 1) * ih) / oh));

    vec4 r = vec4(1.0) / float(ex - sx) / float(ey - sy);
    vec4 acc = vec4(0);

    int xi, yi;
    for (xi = sx; xi < ex; ++xi) {
      for (yi = sy; yi < ey; ++yi) {
        acc += texelFetch(uInput, ivec3(xi, yi, pos.z), 0);
      }
    }

    imageStore(uOutput, pos, r * acc);
  }
}
