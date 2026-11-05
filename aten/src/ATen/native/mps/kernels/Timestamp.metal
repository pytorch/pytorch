// Dispatched alone in the encoder that samples a timing event's GPU timestamp
// (see MPSEvent.mm): Metal skips encoders with no work, and with them their
// samples.
kernel void timestamp_noop(device uint* out [[buffer(0)]]) {
  out[0] = 0;
}
