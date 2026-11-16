/*
 * Random utility functions
 */

// the epsilong defined for fp16 in PyTorch
#define PI 3.14159265358979323846264

uint pcg_hash(uint v) {
  // From: https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float rand2(ivec4 pos) {
  uint s =
      pcg_hash(pos.x) + pcg_hash(pos.y) + pcg_hash(pos.z) + pcg_hash(pos.w);
  return fract(s / 1234567.0);
}

float rand2_nonzero(ivec4 pos) {
  float v = rand2(pos);
  int offset = 0;
  while (v == 0.0) {
    offset++;
    v = rand2(ivec4(pos.x + offset, pos.y, pos.z, pos.w));
  }
  return v;
}

float get_uniform(ivec4 pos, float from, float to) {
  float v = rand2(pos);
  return from + v * (to - from);
}

float get_gaussrand(ivec4 pos, float mean, float std) {
  // Implementation of Box-Muller transform from the pseudo from Wikipedia,
  // which converts two uniformly sampled random numbers into two numbers of
  // Gaussian distribution. Since the shader file can only use one for a
  // position, we flip a coin by the 3rd uniformly sampled number to decide
  // which one to keep.
  // https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  float u1 = rand2_nonzero(pos);
  float u2 =
      rand2_nonzero(ivec4(pos.x + 10, pos.y + 20, pos.z + 30, pos.w + 40));
  float u3 =
      rand2_nonzero(ivec4(pos.x - 10, pos.y - 20, pos.z - 30, pos.w - 40));

  float mag = std * sqrt(-2.0 * log(u1));
  float v;
  if (u3 > 0.5)
    v = mag * cos(2.0 * PI * u2) + mean;
  else
    v = mag * sin(2.0 * PI * u2) + mean;
  return v;
}
