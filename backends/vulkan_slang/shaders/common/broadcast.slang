// Broadcasting logic for binary operations

// Compute broadcast-compatible index
uint broadcast_index(uint flat_idx, uint src_numel, uint dst_numel) {
    // If source is smaller, wrap around (broadcast)
    return flat_idx % src_numel;
}
