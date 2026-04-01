// A regression: the do-while loop ir lowering logic is making incorrect assumptions and is
// broken after short circuiting expr change, leading to invalid IR being generated from lowering
// (a block has two terminator insts).

//TEST:SIMPLE(filecheck=CHECK): -target spirv -emit-spirv-directly

// CHECK: OpEntryPoint

struct Constants {
  uint *buffer;
}
[[vk::push_constant]] Constants push_constants;

[shader("compute")] [numthreads(1, 1, 1)] void compute(void) {
  uint index = push_constants.buffer[0];

  [[unroll]] for (uint i = 0; i < 2; i++) {
    GroupMemoryBarrierWithWaveSync();

    if (index > 0 && i < push_constants.buffer[index - 1]) {
      do {
        index--;
      } while (index > 0 && i < push_constants.buffer[index - 1]);
    }
  }
}
