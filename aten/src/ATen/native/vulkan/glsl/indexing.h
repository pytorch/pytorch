/*
 * Computes a 4D tensor co-ordinate from a linearized index
 */
uvec4 idx_to_coord(const uint idx, const uvec4 strides, const uvec4 sizes) {
  return ivec4(mod(idx / strides, sizes));
}

/*
 * Computes a linearized index from a 4D tensor co-ordinate
 */
uint coord_to_idx(const uvec4 coord, const uvec4 strides) {
  return int(dot(coord * strides, ivec4(1)));
}
