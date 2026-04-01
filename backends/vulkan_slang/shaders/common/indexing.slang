// N-dimensional index <-> linear offset conversion with stride support

// Convert N-dim index to flat offset using strides
uint nd_to_flat_4d(uint4 idx, uint4 strides) {
    return idx.x * strides.x + idx.y * strides.y +
           idx.z * strides.z + idx.w * strides.w;
}

// Convert flat offset to N-dim index (contiguous layout)
uint4 flat_to_nd_4d(uint flat, uint4 sizes) {
    uint4 idx;
    idx.w = flat % sizes.w; flat /= sizes.w;
    idx.z = flat % sizes.z; flat /= sizes.z;
    idx.y = flat % sizes.y; flat /= sizes.y;
    idx.x = flat;
    return idx;
}
