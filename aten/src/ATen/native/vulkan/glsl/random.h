/*
 * Random utility functions
 */

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

float get_uniform(ivec4 pos, float from, float to) {
  float v = rand2(pos);
  return from + v * (to - from);
}
