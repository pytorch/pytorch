// Test cases for process_alloc_data in torch/utils/viz/MemoryViz.js
// Run: node test/profiler/test_memory_viz.js

'use strict';

// Polyfill for Node < 18
if (!Array.prototype.findLastIndex) {
  Array.prototype.findLastIndex = function(pred) {
    for (let i = this.length - 1; i >= 0; i--) {
      if (pred(this[i], i, this)) return i;
    }
    return -1;
  };
}
if (!Array.prototype.at) {
  Array.prototype.at = function(n) {
    return n < 0 ? this[this.length + n] : this[n];
  };
}

// ============================================================
// Load process_alloc_data from the actual MemoryViz.js source
// ============================================================

const fs = require('fs');
const path = require('path');
const vm = require('vm');

// Load process_alloc_data.js (ESM file) in Node.js CommonJS context.
// Strip the `export` line since Node CommonJS doesn't support ESM syntax.
const modPath = path.resolve(__dirname, '../../torch/utils/viz/process_alloc_data.js');
let src = fs.readFileSync(modPath, 'utf-8');
src = src.replace(/^export\s*\{[^}]*\};?\s*$/gm, '');
const wrapper = `(function() { ${src}\nreturn { process_alloc_data, isPrivatePoolId, formatSize, formatAddr, elideRepeats }; })()`;
const { process_alloc_data, isPrivatePoolId, formatSize, formatAddr, elideRepeats } = vm.runInThisContext(wrapper, { filename: modPath });

// ============================================================
// Test helpers
// ============================================================

function makeSnapshot({ traces = [], segments = [], categories = [] }) {
  return {
    device_traces: [traces],
    segments,
    categories,
  };
}

let passed = 0;
let failed = 0;

function assert(condition, msg) {
  if (!condition) {
    failed++;
    console.error(`  FAIL: ${msg}`);
    console.trace();
  } else {
    passed++;
  }
}

function assertEqual(actual, expected, msg) {
  if (actual !== expected) {
    failed++;
    console.error(`  FAIL: ${msg} — expected ${expected}, got ${actual}`);
  } else {
    passed++;
  }
}

function assertContains(str, substr, msg) {
  if (!str.includes(substr)) {
    failed++;
    console.error(`  FAIL: ${msg} — "${substr}" not found in "${str.slice(0, 200)}..."`);
  } else {
    passed++;
  }
}

// ============================================================
// Tests
// ============================================================

function test_basic_alloc_free() {
  console.log('test_basic_alloc_free');
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 1000, size: 100, frames: [], stream: 0 },
      { action: 'free_completed', addr: 1000, size: 100, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0],
      stream: 0, blocks: [],
    }],
  });
  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  assertEqual(result.max_size, 100, 'peak should be 100');
}

function test_free_completed_is_matched() {
  console.log('test_free_completed_is_matched');
  // Only free_completed is matched (not free_requested). Verify alloc+free_completed works.
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 1000, size: 200, frames: [], stream: 0 },
      { action: 'free_completed', addr: 1000, size: 200, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0],
      stream: 0, blocks: [],
    }],
  });
  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  assertEqual(result.max_size, 200, 'alloc+free_completed peak should be 200');
}

function test_pool_free_without_alloc_no_inflation() {
  console.log('test_pool_free_without_alloc_no_inflation');
  // Simulate: a private pool block was allocated BEFORE trace recording,
  // then freed during recording. The trace only has free_completed.
  // With include_private_inactive=true, the block should be pre-loaded into
  // pool state and then freed — NOT treated as a new allocation.
  const poolId = [1, 42];
  const snapshot = makeSnapshot({
    traces: [
      // No alloc — it happened before recording started or got replaced in ring buffer
      // Only the free_completed event is matched (free_requested is ignored).
      { action: 'free_completed', addr: 0x788c2000000, size: 94371840, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x788c2000000, size: 256, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0x788c0000000, total_size: 83886080, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // The free_completed creates an element in initially_allocated + actions.
  // Pre-loaded into pool: pool.active=90M, envelope=max(90M, 80M reserved)=90M.
  // Then freed (pool.active=0), then small alloc (pool.active=256).
  // Peak should be 94371840 (max of active and reserved), NOT double-counted.
  assertEqual(result.max_size, 94371840,
    'pool free-without-alloc peak should be max(active, reserved)');

  assertEqual(result.elements_length, 2, 'should have 2 elements');

  assert(result.max_at_time.length > 0, 'max_at_time should not be empty');
  assertEqual(Math.max(...result.max_at_time), 94371840,
    'max_at_time peak should be max(active, reserved)');

  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 pool envelope');
  assertEqual(envelopes[0].elem, 'pool:1,42,s0', 'envelope key matches pool id and stream');

  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 2, 'should have 2 pool stripes');
  assertEqual(stripes[0].size, 94371840, 'first stripe: pre-loaded block (freed later)');
  assertEqual(stripes[1].size, 256, 'second stripe: small alloc after free');
}

function test_pool_alloc_then_free_normal() {
  console.log('test_pool_alloc_then_free_normal');
  // Normal case: alloc + free both in trace, private pool, include_private_inactive=true
  const poolId = [1, 7];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 2000, size: 500, frames: [], stream: 0 },
      { action: 'free_completed', addr: 2000, size: 500, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 8192, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // Envelope = segment reserved (8192), which is the pool's GPU footprint
  assertEqual(result.max_size, 8192, 'pool envelope should be segment reserved (8192)');
}

function test_multiple_pool_frees_without_alloc() {
  console.log('test_multiple_pool_frees_without_alloc');
  // Multiple blocks freed from the same pool without matching allocs.
  // This is the FSDP scenario with many free_storage calls.
  const poolId = [1, 99];
  const snapshot = makeSnapshot({
    traces: [
      // Each block has one free_completed (free_requested is ignored by the JS code)
      { action: 'free_completed', addr: 1000, size: 500, frames: [], stream: 0 },
      { action: 'free_completed', addr: 2000, size: 500, frames: [], stream: 0 },
      { action: 'free_completed', addr: 3000, size: 500, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 16384, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // 3 blocks of 500 each were initially allocated. Pool envelope = 16384 (segment reserved).
  // Then all 3 are freed. Peak should be 16384 (segment reserved, the initial state).
  // BUG would give 32768+ (double-counted).
  assert(result.max_size <= 16384,
    `multiple pool frees should not inflate: got ${result.max_size}, expected <= 16384`);
}

function test_non_pool_free_without_alloc() {
  console.log('test_non_pool_free_without_alloc');
  // Non-pool block freed without matching alloc (ring buffer wrap).
  // Should appear then disappear.
  const snapshot = makeSnapshot({
    traces: [
      { action: 'free_completed', addr: 8000, size: 300, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 16384, segment_pool_id: [0, 0],
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  // max_size is only updated inside the actions loop AFTER the free decrements total_mem,
  // so for a free-without-alloc element, max_size ends up 0.
  // The actual peak (300) is captured in max_at_time instead.
  assertEqual(result.max_size, 0, 'non-pool free-without-alloc: max_size is 0 (peak is in max_at_time)');
  assert(Math.max(...result.max_at_time) === 300,
    'non-pool free-without-alloc: max_at_time peak should be 300');
}

function test_mixed_pool_and_nonpool() {
  console.log('test_mixed_pool_and_nonpool');
  // Mix of pool and non-pool allocations
  const poolId = [1, 5];
  const snapshot = makeSnapshot({
    traces: [
      // Non-pool alloc+free (addr within default pool segment)
      { action: 'alloc', addr: 100, size: 200, frames: [], stream: 0 },
      // Pool alloc+free (addr within private pool segment)
      { action: 'alloc', addr: 5000, size: 400, frames: [], stream: 0 },
      { action: 'free_completed', addr: 100, size: 200, frames: [], stream: 0 },
      { action: 'free_completed', addr: 5000, size: 400, frames: [], stream: 0 },
    ],
    segments: [
      { device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0], stream: 0, blocks: [] },
      { device: 0, address: 4096, total_size: 8192, segment_pool_id: poolId, stream: 0, blocks: [] },
    ],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // Peak: 200 (non-pool) + 8192 (pool envelope = segment reserved) = 8392
  assertEqual(result.max_size, 8392, 'mixed pool+nonpool peak: 200 + 8192 segment reserved');
}

function test_include_private_inactive_false_ignores_pools() {
  console.log('test_include_private_inactive_false_ignores_pools');
  // When include_private_inactive=false, pool blocks should be treated as regular
  const poolId = [1, 10];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 5000, size: 800, frames: [], stream: 0 },
      { action: 'free_completed', addr: 5000, size: 800, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 4096, total_size: 8192, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  // Without pool logic, it's just a regular alloc+free. Peak = 800.
  assertEqual(result.max_size, 800, 'with include_private_inactive=false, peak = 800');
}

// ============================================================
// formatSize tests
// ============================================================

function test_formatSize_bytes() {
  console.log('test_formatSize_bytes');
  assertEqual(formatSize(0), '0.0B (0 bytes)', 'zero bytes');
  assertEqual(formatSize(512), '512.0B (512 bytes)', '512 bytes');
  assertEqual(formatSize(1), '1.0B (1 bytes)', '1 byte');
}

function test_formatSize_kib() {
  console.log('test_formatSize_kib');
  // 1024 bytes = 1.0 KiB
  assertEqual(formatSize(1024), '1.0KiB (1024 bytes)', '1 KiB');
  // 1536 = 1.5 * 1024
  assertEqual(formatSize(1536), '1.5KiB (1536 bytes)', '1.5 KiB');
}

function test_formatSize_mib_gib() {
  console.log('test_formatSize_mib_gib');
  const mib = 1024 * 1024;
  assertEqual(formatSize(mib), '1.0MiB (1048576 bytes)', '1 MiB');
  const gib = 1024 * 1024 * 1024;
  assertEqual(formatSize(gib), '1.0GiB (1073741824 bytes)', '1 GiB');
}

function test_formatSize_no_bytes() {
  console.log('test_formatSize_no_bytes');
  assertEqual(formatSize(1024, false), '1.0KiB', 'showBytes=false omits raw count');
  assertEqual(formatSize(512, false), '512.0B', 'showBytes=false for small values');
}

// ============================================================
// formatAddr tests
// ============================================================

function test_formatAddr_block_event() {
  console.log('test_formatAddr_block_event');
  const event = { action: 'alloc', addr: 0x7f4c00000, version: 3 };
  assertEqual(formatAddr(event), "b'7f4c00000_3", 'block alloc address');
}

function test_formatAddr_segment_event() {
  console.log('test_formatAddr_segment_event');
  const event = { action: 'segment_alloc', addr: 0xabc, version: 0 };
  assertEqual(formatAddr(event), "s'abc_0", 'segment alloc address');
}

function test_formatAddr_free_event() {
  console.log('test_formatAddr_free_event');
  const event = { action: 'free_completed', addr: 0xff, version: 5 };
  assertEqual(formatAddr(event), "b'ff_5", 'free_completed is a block event');
}

// ============================================================
// elideRepeats tests
// ============================================================

function test_elideRepeats_no_repeats() {
  console.log('test_elideRepeats_no_repeats');
  const result = elideRepeats(['a', 'b', 'c']);
  assertEqual(result.join(','), 'a,b,c', 'no repeats passes through');
}

function test_elideRepeats_two_consecutive() {
  console.log('test_elideRepeats_two_consecutive');
  // Two consecutive duplicates are kept as-is (not collapsed)
  const result = elideRepeats(['a', 'a', 'b']);
  assertEqual(result.join(','), 'a,a,b', 'two consecutive kept verbatim');
}

function test_elideRepeats_three_or_more() {
  console.log('test_elideRepeats_three_or_more');
  // Three+ consecutive duplicates collapse to [frame, "<repeats N times>"]
  const result = elideRepeats(['x', 'x', 'x', 'x']);
  assertEqual(result.length, 2, 'collapsed to 2 entries');
  assertEqual(result[0], 'x', 'first entry is the frame');
  assertEqual(result[1], '<repeats 3 times>', 'second entry is repeat count');
}

function test_elideRepeats_mixed() {
  console.log('test_elideRepeats_mixed');
  // Realistic: a stack trace with a recursive section in the middle
  const result = elideRepeats(['top', 'recurse', 'recurse', 'recurse', 'recurse', 'recurse', 'bottom']);
  assertEqual(result.join(','), 'top,recurse,<repeats 4 times>,bottom', 'mixed with recursion');
}

function test_elideRepeats_empty() {
  console.log('test_elideRepeats_empty');
  const result = elideRepeats([]);
  assertEqual(result.length, 0, 'empty input gives empty output');
}

// ============================================================
// context_for_id tests
// ============================================================

function test_context_for_id_with_pool() {
  console.log('test_context_for_id_with_pool');
  // Alloc with a known private pool, stack frames, and stream
  const poolId = [2, 7];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 0xabc000, size: 2048, version: 5,
        frames: [{ filename: 'model.py', line: 42, name: 'forward' }],
        stream: 3, timestamp: true, time_us: 1700000000000000 },
    ],
    segments: [{
      device: 0, address: 0xa00000, total_size: 0x200000, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  const ctx = result.context_for_id(0);

  // Address should be formatted as hex with version
  assertContains(ctx, "abc000", 'context includes hex address');
  assertContains(ctx, '_5', 'context includes version');
  // Size should be formatted
  assertContains(ctx, '2.0KiB', 'context includes formatted size');
  assertContains(ctx, '2048 bytes', 'context includes raw byte count');
  // Pool ID resolved from segment
  assertContains(ctx, 'pool_id (2, 7)', 'context includes resolved pool_id');
  // Stream
  assertContains(ctx, 'stream 3', 'context includes stream');
  // Stack frame
  assertContains(ctx, 'model.py:42:forward', 'context includes stack frame');
}

function test_context_for_id_unknown_pool() {
  console.log('test_context_for_id_unknown_pool');
  // Alloc at address outside any segment → pool_id unknown
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 0xff0000, size: 512, version: 0,
        frames: [{ filename: 'train.py', line: 10, name: 'step' }],
        stream: 0, timestamp: null },
    ],
    segments: [{
      device: 0, address: 0x100000, total_size: 0x100000, segment_pool_id: [0, 0],
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  const ctx = result.context_for_id(0);

  assertContains(ctx, 'pool_id unknown', 'addr outside segments gives unknown pool');
  assertContains(ctx, 'ff0000', 'context includes hex address');
  assertContains(ctx, '512.0B', 'context includes size');
}

function test_context_for_id_free_without_alloc() {
  console.log('test_context_for_id_free_without_alloc');
  // free_completed without matching alloc → "alloc not recorded" message
  const snapshot = makeSnapshot({
    traces: [
      { action: 'free_completed', addr: 0xdead00, size: 4096, version: 2,
        frames: [{ filename: 'fsdp.py', line: 750, name: 'free_storage' }],
        stream: 0, timestamp: null },
    ],
    segments: [{
      device: 0, address: 0xd00000, total_size: 0x100000, segment_pool_id: [0, 0],
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  const ctx = result.context_for_id(0);

  assertContains(ctx, 'alloc not recorded', 'free-without-alloc shows warning');
  assertContains(ctx, 'dead00', 'context includes hex address');
  assertContains(ctx, '4.0KiB', 'context includes size');
  assertContains(ctx, 'fsdp.py:750:free_storage', 'context shows free stack trace');
}

// ============================================================
// Post-PR#177717 tests: trace events carry segment_pool_id directly
// ============================================================

function test_post177717_pool_id_from_trace_event() {
  console.log('test_post177717_pool_id_from_trace_event');
  // After PR#177717, trace events include segment_pool_id. The code should
  // use it directly instead of falling back to find_pool_id from segments.
  // Here the addr is OUTSIDE any segment, but pool_id is on the event itself.
  const poolId = [3, 15];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 0xff0000, size: 1024, frames: [], stream: 0,
        segment_pool_id: poolId },
      { action: 'free_completed', addr: 0xff0000, size: 1024, frames: [], stream: 0,
        segment_pool_id: poolId },
    ],
    // No segment covers 0xff0000 — pool_id comes from the trace event
    segments: [],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  assertEqual(result.max_size, 1024, 'pool alloc/free with event-level pool_id');

  const ctx = result.context_for_id(0);
  assertContains(ctx, 'pool_id (3, 15)', 'pool_id resolved from trace event, not segment');
}

function test_post177717_pool_free_without_alloc_no_segment() {
  console.log('test_post177717_pool_free_without_alloc_no_segment');
  // Post-177717: free_completed has segment_pool_id on the event.
  // The segment was unmapped (not in segments list), but pool_id is still known.
  const poolId = [1, 42];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'free_completed', addr: 0xdead00, size: 2048, frames: [], stream: 0,
        segment_pool_id: poolId },
    ],
    // Segment was unmapped — not present. Pool resolved from event.
    segments: [],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // Should be pre-loaded into pool and then freed, not inflated
  assert(result.max_size <= 2048,
    `post-177717 pool free-without-alloc should not inflate: got ${result.max_size}`);

  const ctx = result.context_for_id(0);
  assertContains(ctx, 'pool_id (1, 42)', 'pool_id from event even without segment');
  assertContains(ctx, 'alloc not recorded', 'still shows alloc not recorded');
}

function test_post177717_mixed_events_with_and_without_pool_id() {
  console.log('test_post177717_mixed_events_with_and_without_pool_id');
  // Some events have segment_pool_id (post-177717), others don't (pre-177717
  // or default pool). Verify both paths work together.
  const poolId = [2, 8];
  const snapshot = makeSnapshot({
    traces: [
      // Default pool alloc — no segment_pool_id on event, resolved via segment
      { action: 'alloc', addr: 100, size: 300, frames: [], stream: 0 },
      // Private pool alloc — segment_pool_id on the event
      { action: 'alloc', addr: 0xf000, size: 500, frames: [], stream: 0,
        segment_pool_id: poolId },
      { action: 'free_completed', addr: 100, size: 300, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0xf000, size: 500, frames: [], stream: 0,
        segment_pool_id: poolId },
    ],
    segments: [
      // Only covers addr=100 (default pool). addr=0xf000 has no segment.
      { device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0],
        stream: 0, blocks: [] },
    ],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // Peak: 300 (non-pool) + 500 (pool envelope) = 800
  assertEqual(result.max_size, 800, 'mixed pre/post-177717 events peak');

  const ctx0 = result.context_for_id(0);
  assertContains(ctx0, 'pool_id (0, 0)', 'default pool resolved from segment');

  const ctx1 = result.context_for_id(1);
  assertContains(ctx1, 'pool_id (2, 8)', 'private pool from event-level segment_pool_id');
}


// ============================================================
// Pool grouping by (pool_id, stream) tests
// ============================================================

function test_pool_grouped_by_stream() {
  console.log('test_pool_grouped_by_stream');
  // Same pool_id but different streams should produce separate envelopes.
  const poolId = [1, 5];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 1000, size: 500, frames: [], stream: 1 },
      { action: 'alloc', addr: 2000, size: 300, frames: [], stream: 2 },
      { action: 'free_completed', addr: 1000, size: 500, frames: [], stream: 1 },
      { action: 'free_completed', addr: 2000, size: 300, frames: [], stream: 2 },
    ],
    segments: [
      { device: 0, address: 0, total_size: 4096, segment_pool_id: poolId,
        stream: 0, blocks: [] },
    ],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);

  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 2, 'should have 2 pool envelopes (one per stream)');

  const keys = envelopes.map(e => e.elem).sort();
  assertEqual(keys[0], 'pool:1,5,s1', 'first envelope key includes stream 1');
  assertEqual(keys[1], 'pool:1,5,s2', 'second envelope key includes stream 2');
}

// ============================================================
// Initially Added Blocks Tests
// ============================================================

function test_default_pool_ghost_block() {
  console.log('test_segment_snapshot_no_trace');
  const poolId = [0, 0];
  const snapshot = makeSnapshot({
    traces: [],
    segments: [{
      device: 0, address: 4096, total_size: 8192, segment_pool_id: poolId,
      stream: 0, blocks: [
        { address: 5000, size: 1000, requested_size: 1000, state: 'active_allocated',
          frames: [], version: 0 },
      ],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);
  assertEqual(result.elements_length, 1,
    'snapshot-only block should not be added (include_private_inactive=false)');

}

function test_segment_snapshot_with_trace_history() {
  console.log('test_segment_snapshot_with_trace_history');
  const poolId = [1, 42];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 5000, size: 1000, frames: [], stream: 0 },
      { action: 'free_completed', addr: 5000, size: 1000, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 4096, total_size: 8192, segment_pool_id: poolId,
      stream: 0, blocks: [
        { address: 5000, size: 1000, requested_size: 1000, state: 'inactive',
          frames: [], version: 0 },
      ],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  assertEqual(result.elements_length, 1,
    'only trace element present, snapshot block not duplicated');
}

function test_segment_snapshot_no_trace() {
  console.log('test_segment_snapshot_no_trace');
  const poolId = [1, 42];
  const snapshot = makeSnapshot({
    traces: [],
    segments: [{
      device: 0, address: 4096, total_size: 8192, segment_pool_id: poolId,
      stream: 0, blocks: [
        { address: 5000, size: 1000, requested_size: 1000, state: 'inactive',
          frames: [], version: 0 },
      ],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  assertEqual(result.elements_length, 0,
    'snapshot-only block should not be added (include_private_inactive=true)');

  const result2 = process_alloc_data(snapshot, 0, false, 15000, false);
  assertEqual(result2.elements_length, 0,
    'snapshot-only block should not be added (include_private_inactive=false)');
}

function test_ghost_blocks() {
  console.log('test_ghost_blocks');
  // Snapshot produced by (agent_space/test_ring_buffer_overflow.py):
  //   pre_record = torch.empty(1024 * 1024, device="cuda", dtype=torch.uint8)  # 1 MiB
  //   torch.cuda.memory._record_memory_history(max_entries=10)
  //   early = torch.empty(2 * 1024 * 1024, device="cuda", dtype=torch.uint8)   # 2 MiB
  //   for _ in range(15):  # overflow the 10-entry ring buffer
  //     t = torch.empty(4 * 1024 * 1024, device="cuda", dtype=torch.uint8)     # 4 MiB
  //     del t
  //   snap = torch.cuda.memory._snapshot()
  //
  // pre_record: allocated before recording → no trace event at all
  // early: alloc event evicted from ring buffer by churn → no trace event
  // Both are active_allocated in segment snapshot but invisible in trace.
  const snapshot = makeSnapshot({
    traces: [
      { action: 'free_completed', addr: 0x6a00000, size: 4194304, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x6a00000, size: 4194304, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x6a00000, size: 4194304, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x6a00000, size: 4194304, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x6a00000, size: 4194304, frames: [], stream: 0 },
    ],
    segments: [
      { device: 0, address: 0x1e00000, total_size: 2097152, segment_pool_id: [0, 0],
        stream: 0, blocks: [
          { address: 0x1e00000, size: 1048576, requested_size: 1048576,
            state: 'active_allocated', frames: [] },
          { address: 0x1f00000, size: 1048576, requested_size: 1048576,
            state: 'inactive', frames: [] },
        ]},
      { device: 0, address: 0x6800000, total_size: 20971520, segment_pool_id: [0, 0],
        stream: 0, blocks: [
          { address: 0x6800000, size: 2097152, requested_size: 2097152,
            state: 'active_allocated', frames: [
              { filename: 'test.py', line: 10, name: 'early_alloc' },
            ]},
          { address: 0x6a00000, size: 18874368, requested_size: 18874368,
            state: 'inactive', frames: [] },
        ]},
    ],
  });

  // Ghost blocks (active_allocated not in trace) show on both tabs.
  for (const include_private of [false, true]) {
    const label = `include_private_inactive=${include_private}`;
    const result = process_alloc_data(snapshot, 0, false, 15000, include_private);

    // Trace creates 2 elements from alloc events + 1 from unmatched free.
    // 2 active_allocated blocks from snapshot not in trace.
    assertEqual(result.elements_length, 5,
      `${label}: 3 trace elements + 2 snapshot blocks`);

    const aot = result.allocations_over_time;
    const ghosts = aot.filter(d => d.ghost === true);
    assertEqual(ghosts.length, 2, `${label}: should have 2 ghost block entries`);

    // Ghost block sizes match segment snapshot blocks
    const ghost_sizes = ghosts.map(g => g.size).sort();
    assertEqual(ghost_sizes[0], 1048576, `${label}: ghost block 1 MiB (pre_record)`);
    assertEqual(ghost_sizes[1], 2097152, `${label}: ghost block 2 MiB (ring buffer overflow)`);

    // context_for_id shows ghost explanation
    const ghost_elem_ids = ghosts.map(g => g.elem);
    for (const id of ghost_elem_ids) {
      const ctx = result.context_for_id(id);
      assertContains(ctx, '[Ghost block]', `${label}: context contains ghost label`);
      assertContains(ctx, 'segment snapshot', `${label}: context explains source`);
    }
  }
}

function test_ghost_blocks_not_created_for_traced_addrs() {
  console.log('test_ghost_blocks_not_created_for_traced_addrs');
  // A block in the segment snapshot whose address DID appear in trace events
  // should NOT be a ghost block.
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 1000, size: 512, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0],
      stream: 0, blocks: [
        { address: 1000, size: 512, requested_size: 512,
          state: 'active_allocated', frames: [] },
      ],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  const aot = result.allocations_over_time;
  const ghosts = aot.filter(d => d.ghost === true);
  assertEqual(ghosts.length, 0, 'no ghost blocks when addr is in trace');
  assertEqual(result.elements_length, 1, 'only the trace element');
}

function test_ghost_blocks_default_pool_collected() {
  console.log('test_ghost_blocks_default_pool_collected');
  // Ghost blocks from default pool [0,0] should be collected
  // from the segment snapshot when they have no trace events.
  const snapshot = makeSnapshot({
    traces: [],
    segments: [{
      device: 0, address: 0x1000, total_size: 8192, segment_pool_id: [0, 0],
      stream: 0, blocks: [
        { address: 0x1000, size: 2048, requested_size: 2048,
          state: 'active_allocated', frames: [] },
        { address: 0x1800, size: 1024, requested_size: 1024,
          state: 'active_allocated', frames: [] },
      ],
    }],
  });

  // Shows on both tabs
  for (const include_private of [false, true]) {
    const label = `include_private_inactive=${include_private}`;
    const result = process_alloc_data(snapshot, 0, false, 15000, include_private);
    const ghosts = result.allocations_over_time.filter(d => d.ghost === true);
    assertEqual(ghosts.length, 2, `${label}: 2 ghost blocks from default pool`);
    const sizes = ghosts.map(g => g.size).sort();
    assertEqual(sizes[0], 1024, `${label}: ghost 1024 bytes`);
    assertEqual(sizes[1], 2048, `${label}: ghost 2048 bytes`);
  }
}

function test_ghost_blocks_not_in_segment_mode() {
  console.log('test_ghost_blocks_not_in_segment_mode');
  // Ghost blocks should not be created in segment-level views
  const snapshot = makeSnapshot({
    traces: [],
    segments: [{
      device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0],
      stream: 0, blocks: [
        { address: 100, size: 512, requested_size: 512,
          state: 'active_allocated', frames: [] },
      ],
    }],
  });

  const result_seg = process_alloc_data(snapshot, 0, true, 15000, false);
  const ghosts_seg = result_seg.allocations_over_time.filter(d => d.ghost === true);
  assertEqual(ghosts_seg.length, 0, 'no ghost blocks in segment_alloc mode');

  // Even with include_private_inactive=true, segment modes should not create ghosts
  const result_seg2 = process_alloc_data(snapshot, 0, true, 15000, true);
  const ghosts_seg2 = result_seg2.allocations_over_time.filter(d => d.ghost === true);
  assertEqual(ghosts_seg2.length, 0, 'no ghost blocks in segment_alloc mode (private pool tab)');
}

function test_ghost_blocks_private_pool() {
  console.log('test_ghost_blocks_private_pool');
  // Snapshot produced by (agent_space/test_ghost_blocks_private_pool.py):
  //   pool = torch.cuda.MemPool()
  //   with torch.cuda.use_mem_pool(pool):
  //     pre_record_pool = torch.empty(1 MiB)     # ghost in private pool (0,1)
  //   pre_record_default = torch.empty(2 MiB)     # ghost in default pool (0,0)
  //   torch.cuda.memory._record_memory_history(max_entries=20)
  //   with torch.cuda.use_mem_pool(pool):
  //     traced_pool = torch.empty(3 MiB)          # traced in private pool (0,1)
  //   traced_default = torch.empty(4 MiB)          # traced in default pool (0,0)
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 0xe600000, size: 3145728, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x6a00000, size: 4194304, frames: [], stream: 0 },
    ],
    segments: [
      // Private pool segment with ghost block
      { device: 0, address: 0x1e00000, total_size: 2097152, segment_pool_id: [0, 1],
        stream: 0, blocks: [
          { address: 0x1e00000, size: 1048576, requested_size: 1048576,
            state: 'active_allocated', frames: [] },
        ]},
      // Default pool segment with ghost block + traced block
      { device: 0, address: 0x6800000, total_size: 20971520, segment_pool_id: [0, 0],
        stream: 0, blocks: [
          { address: 0x6800000, size: 2097152, requested_size: 2097152,
            state: 'active_allocated', frames: [] },
          { address: 0x6a00000, size: 4194304, requested_size: 4194304,
            state: 'active_allocated', frames: [] },
        ]},
      // Private pool segment with traced block
      { device: 0, address: 0xe600000, total_size: 20971520, segment_pool_id: [0, 1],
        stream: 0, blocks: [
          { address: 0xe600000, size: 3145728, requested_size: 3145728,
            state: 'active_allocated', frames: [] },
        ]},
    ],
  });

  // With include_private_inactive=true, ghost blocks from BOTH pools are
  // collected. Private pool ghosts go inside their pool envelope; default
  // pool ghosts are rendered at the global bottom of the stacked area.
  const result = process_alloc_data(snapshot, 0, false, 15000, true);

  const aot = result.allocations_over_time;
  const ghosts = aot.filter(d => d.ghost === true);
  assertEqual(ghosts.length, 2, '2 ghost blocks (one from each pool)');

  // Default pool ghost (2 MiB): at global bottom, spans full timeline
  const default_ghost = ghosts.find(g => g.size === 2097152);
  assert(default_ghost !== undefined, 'default pool ghost (2 MiB) exists');
  assertEqual(default_ghost.offsets[0], 0, 'default ghost at offset 0');
  assertEqual(default_ghost.timesteps[0], 0, 'default ghost starts at timestep 0');
  assertEqual(default_ghost.timesteps.length, 2, 'default ghost has 2 timesteps');
  assert(default_ghost.timesteps[1] > 0, 'default ghost ends after timestep 0');

  // Private pool ghost (1 MiB): inside the pool (0,1) envelope
  const pool_ghost = ghosts.find(g => g.size === 1048576);
  assert(pool_ghost !== undefined, 'private pool ghost (1 MiB) exists');
  assertEqual(pool_ghost.timesteps[0], 0, 'pool ghost starts at timestep 0');
  assert(pool_ghost.timesteps.at(-1) > 0, 'pool ghost ends after timestep 0');

  // Both ghosts should end at the same final timestep
  assertEqual(default_ghost.timesteps[1], pool_ghost.timesteps.at(-1),
    'both ghosts end at the same final timestep');

  // Pool envelope only for (0,1) — default pool ghosts are at global bottom.
  // Both pool segments use stream 0, so there should be exactly 1 envelope
  // (not split by stream, and not "snull" from annotate_snapshot eliding streams).
  const envelopes = aot.filter(d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, '1 pool envelope');
  assertContains(envelopes[0].elem, '0,1,s0', 'envelope for pool (0,1)');

  // Envelope initial size at timestep 0 = total pool reserved from snapshot
  // (ghost segment 2 MiB + traced segment ~20 MiB = ~22 MiB)
  const pool_reserved = 2097152 + 20971520;  // sum of pool (0,1) segment total_sizes
  const env = envelopes[0];
  assertEqual(env.timesteps[0], 0, 'envelope starts at timestep 0');
  assertEqual(env.size[0], pool_reserved, 'envelope initial size = total pool reserved');

  // Ghost stripe should fit within the envelope
  const env_offset = env.offsets[0];
  const ghost_offset = pool_ghost.offsets[0];
  assert(ghost_offset >= env_offset, 'ghost stripe offset >= envelope offset');
  assert(ghost_offset + 1048576 <= env_offset + env.size[0],
    'ghost stripe fits within envelope at timestep 0');

  // Pool envelope max = total pool reserved (same as initial since no segment events)
  const env_max_size = Array.isArray(env.size)
    ? Math.max(...env.size)
    : env.size;
  assertEqual(env_max_size, pool_reserved, 'pool envelope max = total pool reserved');

  // With include_private_inactive=false, ghosts still exist but no pool envelope
  const result_false = process_alloc_data(snapshot, 0, false, 15000, false);
  const ghosts_false = result_false.allocations_over_time.filter(d => d.ghost === true);
  assertEqual(ghosts_false.length, 2, '2 ghost blocks on regular tab too');
  const envs_false = result_false.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envs_false.length, 0, 'no pool envelopes when include_private_inactive=false');
}

function test_ghost_stripe_offset_with_multiple_pools() {
  console.log('test_ghost_stripe_offset_with_multiple_pools');
  // When multiple private pools have initially_allocated blocks, pool envelopes
  // are stacked. A ghost stripe must have its offset within its own envelope,
  // not at the offset from when the stripe was first created (before other
  // pools shifted it upward).
  const snapshot = makeSnapshot({
    traces: [
      // Traced alloc in default pool so actions is non-empty
      { action: 'alloc', addr: 0x100, size: 100, frames: [], stream: 0 },
    ],
    segments: [
      // Default pool segment for the traced alloc
      { device: 0, address: 0, total_size: 4096, segment_pool_id: [0, 0],
        stream: 0, blocks: [] },
      // Pool (0,3) segment 1: one ghost block.
      // Added to initially_allocated first among private pools.
      { device: 0, address: 0x4000, total_size: 4096, segment_pool_id: [0, 3],
        stream: 0, blocks: [
          { address: 0x4000, size: 3000, requested_size: 3000,
            state: 'active_allocated', frames: [] },
        ]},
      // Pool (0,2): ghost block. Added to initially_allocated second.
      { device: 0, address: 0x10000, total_size: 8192, segment_pool_id: [0, 2],
        stream: 0, blocks: [
          { address: 0x10000, size: 2000, requested_size: 2000,
            state: 'active_allocated', frames: [] },
        ]},
      // Pool (0,3) segment 2: another ghost block.
      // Added to initially_allocated LAST. After reverse(), processed FIRST.
      // This creates pool (0,3) at offset 0. Then pool (0,2) is processed
      // (stripe at offset 1000). Then pool (0,3) segment 1's ghost block
      // grows pool (0,3) from 1000 to 4000, shifting pool (0,2)'s envelope
      // up by 3000 but (without fix) not its stripe.
      { device: 0, address: 0x5000, total_size: 4096, segment_pool_id: [0, 3],
        stream: 0, blocks: [
          { address: 0x5000, size: 1000, requested_size: 1000,
            state: 'active_allocated', frames: [] },
        ]},
    ],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  const aot = result.allocations_over_time;

  const envelopes = aot.filter(d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 2, '2 pool envelopes');

  // Find the pool (0,2) envelope and its ghost stripe
  const env02 = envelopes.find(e => e.elem.includes('0,2'));
  assert(env02 !== undefined, 'pool (0,2) envelope exists');

  const ghosts = aot.filter(d => d.ghost === true);
  assertEqual(ghosts.length, 3, '3 ghost blocks (2 in pool 0,3 + 1 in pool 0,2)');
  // The ghost for pool (0,2) is the 2000-byte one
  const ghost02 = ghosts.find(g => g.size === 2000);
  assert(ghost02 !== undefined, 'pool (0,2) ghost exists');

  // Ghost stripe offset must be within the envelope range
  const env_offset = env02.offsets[0];
  const env_size = env02.size[0];
  const ghost_offset = ghost02.offsets[0];
  assertEqual(ghost_offset, env_offset,
    `ghost offset should equal envelope offset (single block in pool)`);
  assert(ghost_offset + 2000 <= env_offset + env_size,
    `ghost fits within envelope: ${ghost_offset}+2000 <= ${env_offset}+${env_size}`);
}

// ============================================================
// Full snapshot integration test
// ============================================================

function test_full_snapshot_private_pools() {
  console.log('test_full_snapshot_private_pools');
  // 512 KiB allocs so they're visible relative to the 2 MiB envelope
  const S = 524288;
  const snapshot = makeSnapshot({
    traces: [
      { action: 'segment_alloc', addr: 0x7f08cde00000, size: 2097152, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08cde00000, size: S, frames: [], stream: 0 },
      { action: 'segment_alloc', addr: 0x7f08d2800000, size: 2097152, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08d2800000, size: S, frames: [], stream: 0 },
      { action: 'free_requested', addr: 0x7f08cde00000, size: S, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x7f08cde00000, size: S, frames: [], stream: 0 },
      { action: 'free_requested', addr: 0x7f08d2800000, size: S, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x7f08d2800000, size: S, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08cde00000, size: S, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08d2800000, size: S, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08d2880000, size: S, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08cde80000, size: S, frames: [], stream: 0 },
    ],
    segments: [
      { device: 0, address: 0x7f08cde00000, total_size: 2097152,
        segment_pool_id: [0, 0], stream: 0, blocks: [] },
      { device: 0, address: 0x7f08d2800000, total_size: 2097152,
        segment_pool_id: [0, 1], stream: 0, blocks: [] },
    ],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);

  assertEqual(result.elements_length, 6, 'should have 6 elements');
  assertEqual(result.max_size, 2 * S + 2097152, 'peak = 2 default pool blocks + 2M envelope');

  // Timeline (envelope = 2M from segment reserved, never shrinks):
  //   alloc 512K in default pool       → 512K
  //   alloc 512K in private pool       → 512K + 2M envelope = 2.5M
  //   free 512K from default pool      → 2M (envelope stays)
  //   free 512K stripe in pool         → 2M (stripe gone, envelope stays)
  //   alloc 512K in default pool       → 2M + 512K = 2.5M
  //   alloc 512K stripe in pool        → 2.5M (within envelope)
  //   alloc 512K stripe in pool        → 2.5M (within envelope)
  //   alloc 512K in default pool       → 2M + 2*512K = 3M
  const expected_max_at_time = [
    S, S+2097152, S+2097152, S+2097152, S+2097152, S+2097152,
    S+2097152, S+2097152, S+2097152,
    2097152, S+2097152, S+2097152, S+2097152, 2*S+2097152,
  ];
  assertEqual(result.max_at_time.length, expected_max_at_time.length,
    'max_at_time length');
  for (let i = 0; i < expected_max_at_time.length; i++) {
    assertEqual(result.max_at_time[i], expected_max_at_time[i],
      `max_at_time[${i}]`);
  }

  const aot = result.allocations_over_time;

  const envelopes = aot.filter(d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 pool envelope');
  assertEqual(envelopes[0].elem, 'pool:0,1,s0', 'envelope key matches pool (0,1)');

  // Envelope size is driven by segment reserved (2 MiB), not just active allocs
  const env_max = Array.isArray(envelopes[0].size)
    ? Math.max(...envelopes[0].size) : envelopes[0].size;
  assertEqual(env_max, 2097152, 'envelope = segment reserved (2 MiB)');

  const stripes = aot.filter(d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 3, 'should have 3 pool stripes (elements 1, 3, 4)');

  const non_pool = aot.filter(d => typeof d.elem === 'number' && d.opacity === undefined);
  assertEqual(non_pool.length, 3, 'should have 3 non-pool elements (0, 2, 5)');

  const ctx0 = result.context_for_id(0);
  assertContains(ctx0, '7f08cde00000', 'element 0 addr');
  assertContains(ctx0, 'pool_id (0, 0)', 'element 0 pool');

  const ctx1 = result.context_for_id(1);
  assertContains(ctx1, '7f08d2800000', 'element 1 addr');
  assertContains(ctx1, 'pool_id (0, 1)', 'element 1 pool');
}

function test_full_snapshot_no_private_pools() {
  console.log('test_full_snapshot_no_private_pools');
  const snapshot = makeSnapshot({
    traces: [
      { action: 'segment_alloc', addr: 0x7f08cde00000, size: 2097152, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08cde00000, size: 1024, frames: [], stream: 0 },
      { action: 'segment_alloc', addr: 0x7f08d2800000, size: 2097152, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08d2800000, size: 1024, frames: [], stream: 0 },
      { action: 'free_requested', addr: 0x7f08cde00000, size: 1024, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x7f08cde00000, size: 1024, frames: [], stream: 0 },
      { action: 'free_requested', addr: 0x7f08d2800000, size: 1024, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x7f08d2800000, size: 1024, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08cde00000, size: 1024, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08d2800000, size: 1024, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08d2800400, size: 1024, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x7f08cde00400, size: 1024, frames: [], stream: 0 },
    ],
    segments: [
      { device: 0, address: 0x7f08cde00000, total_size: 2097152,
        segment_pool_id: [0, 0], stream: 0, blocks: [] },
      { device: 0, address: 0x7f08d2800000, total_size: 2097152,
        segment_pool_id: [0, 1], stream: 0, blocks: [] },
    ],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, false);

  assertEqual(result.elements_length, 6, 'should have 6 elements');
  assertEqual(result.max_size, 4096, 'peak memory should be 4096');

  const expected_max_at_time = [
    1024, 2048, 2048, 2048, 2048, 2048,
    1024, 1024, 2048, 3072, 4096,
  ];
  assertEqual(result.max_at_time.length, expected_max_at_time.length,
    'max_at_time length');
  for (let i = 0; i < expected_max_at_time.length; i++) {
    assertEqual(result.max_at_time[i], expected_max_at_time[i],
      `max_at_time[${i}]`);
  }

  const aot = result.allocations_over_time;
  const envelopes = aot.filter(d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 0, 'no pool envelopes');

  const stripes = aot.filter(d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 0, 'no pool stripes');

  const regular = aot.filter(d => typeof d.elem === 'number');
  assertEqual(regular.length, 6, 'all 6 elements are regular');

  const ctx0 = result.context_for_id(0);
  assertContains(ctx0, '7f08cde00000', 'element 0 addr');
  assertContains(ctx0, 'Total memory used after allocation: 1.0KiB', 'element 0 total');
  assertContains(ctx0, 'pool_id (0, 0)', 'element 0 pool');
}

// ============================================================
// Pool envelope reserved-memory tests
// ============================================================

function test_envelope_grows_on_segment_map() {
  console.log('test_envelope_grows_on_segment_map');
  // When a private pool alloc triggers segment_map (fragmentation), the envelope
  // should grow to the reserved size, not just the active allocation size.
  // Scenario: alloc 500, free 500, alloc 400 triggers segment_map of 400
  // (can't reuse fragmented free blocks). Reserved = 500 + 400 = 900.
  const poolId = [1, 10];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'segment_map', addr: 0x1000, size: 500, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x1000, size: 500, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0x1000, size: 500, frames: [], stream: 0 },
      { action: 'segment_map', addr: 0x1200, size: 400, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x1200, size: 400, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0x1000, total_size: 900, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // Envelope should be 900 (reserved), not 500 (peak active)
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 pool envelope');
  const env_max = Array.isArray(envelopes[0].size)
    ? Math.max(...envelopes[0].size) : envelopes[0].size;
  assertEqual(env_max, 900, 'envelope should be 900 (reserved), not 500 (peak active)');

  // Stripes inside the envelope should reflect individual block sizes (500, 400),
  // not grow to the envelope size.
  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 2, 'should have 2 pool stripes');
  const stripe_sizes = stripes.map(s => s.size).sort();
  assertEqual(stripe_sizes[0], 400, 'stripe for second alloc = 400');
  assertEqual(stripe_sizes[1], 500, 'stripe for first alloc = 500');
}

function test_envelope_from_initial_reserved() {
  console.log('test_envelope_from_initial_reserved');
  // Segment reserved is 2000 but only 300 is actively allocated.
  // No segment events in trace — initial reserved = snapshot reserved.
  // Envelope should be 2000 (reserved), not 300 (active).
  const poolId = [1, 20];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 0x2000, size: 300, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0x2000, total_size: 2000, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 envelope');
  const env_max = Array.isArray(envelopes[0].size)
    ? Math.max(...envelopes[0].size) : envelopes[0].size;
  assertEqual(env_max, 2000, 'envelope should be 2000 (segment reserved)');
}

function test_envelope_segment_map_no_double_count() {
  console.log('test_envelope_segment_map_no_double_count');
  // Segment_map events in trace + snapshot reserved should not double-count.
  // Trace has segment_map of 600. Snapshot total = 600. So initial reserved = 0.
  // Envelope grows to 600 from the segment_map event.
  const poolId = [1, 30];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'segment_map', addr: 0x3000, size: 600, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x3000, size: 600, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0x3000, total_size: 600, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 envelope');
  const env_max = Array.isArray(envelopes[0].size)
    ? Math.max(...envelopes[0].size) : envelopes[0].size;
  // Should be 600, not 1200 (double-counted)
  assertEqual(env_max, 600, 'envelope should be 600 (no double count)');
}

function test_envelope_active_exceeds_reserved() {
  console.log('test_envelope_active_exceeds_reserved');
  // Edge case: active > reserved (e.g., segment events lost from ring buffer).
  // Envelope should use max(active, reserved).
  // Scenario: segment_map of 500 in trace, snapshot total = 800.
  // initial reserved = 800 - 500 = 300. After segment_map, reserved = 800.
  // Then alloc of 800 → active=800 = reserved=800 → envelope=800.
  // Now a second alloc of 200 with no segment event → active=1000 > reserved=800.
  // Envelope should grow to 1000 (active exceeds reserved).
  const poolId = [1, 40];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'segment_map', addr: 0x4000, size: 500, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x4000, size: 800, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x4400, size: 200, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0x4000, total_size: 1000, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 envelope');
  const env_max = Array.isArray(envelopes[0].size)
    ? Math.max(...envelopes[0].size) : envelopes[0].size;
  // active=1000 > reserved at time of second alloc (800), envelope = 1000
  assertEqual(env_max, 1000, 'envelope should be 1000 (active exceeds earlier reserved)');
}

function test_envelope_default_pool_unaffected() {
  console.log('test_envelope_default_pool_unaffected');
  // Default pool (0,0) should NOT get envelope treatment regardless of segment events.
  const snapshot = makeSnapshot({
    traces: [
      { action: 'segment_map', addr: 100, size: 5000, frames: [], stream: 0 },
      { action: 'alloc', addr: 100, size: 200, frames: [], stream: 0 },
      { action: 'free_completed', addr: 100, size: 200, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0, total_size: 5000, segment_pool_id: [0, 0],
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 0, 'default pool should not have envelope');
  assertEqual(result.max_size, 200, 'default pool peak = active only');
}

// ============================================================
// Per-pool summarization tests
// ============================================================

function test_per_pool_summarization() {
  console.log('test_per_pool_summarization');
  // 10 allocs in default pool (sizes 100..1000), 10 allocs in private pool
  // (sizes 200..2000). Limit to 5 entries per pool.
  // Expect: 5 drawn + 5 summarized per pool.
  const poolId = [1, 50];
  const traces = [];
  // Default pool: 10 allocs at 0x1000..0x1900, sizes 100,200,...,1000
  for (let i = 0; i < 10; i++) {
    traces.push({ action: 'alloc', addr: 0x1000 + i * 0x100, size: (i + 1) * 100,
                  frames: [], stream: 0 });
  }
  // Private pool: 10 allocs at 0x5000..0x5900, sizes 200,400,...,2000
  for (let i = 0; i < 10; i++) {
    traces.push({ action: 'alloc', addr: 0x5000 + i * 0x100, size: (i + 1) * 200,
                  frames: [], stream: 0 });
  }

  const snapshot = makeSnapshot({
    traces,
    segments: [
      { device: 0, address: 0x1000, total_size: 0x1000, segment_pool_id: [0, 0],
        stream: 0, blocks: [] },
      { device: 0, address: 0x5000, total_size: 0x1000, segment_pool_id: poolId,
        stream: 0, blocks: [] },
    ],
  });

  // Global top 5: the 5 largest across all pools.
  // Private pool sizes: 200,400,...,2000. Default pool sizes: 100,200,...,1000.
  // Top 5 globally = 2000,1800,1600,1400,1200 (all from private pool).
  // Default pool: all 10 go to global summarized band.
  // Private pool: top 5 drawn, bottom 5 (200+400+600+800+1000=3000) in per-pool summary.
  const result = process_alloc_data(snapshot, 0, false, 5, true);

  const non_pool = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === undefined);
  assertEqual(non_pool.length, 0, 'default pool: 0 drawn (all smaller than top 5)');

  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 5, 'private pool: 5 drawn stripes');

  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, '1 pool envelope');

  const pool_summaries = result.allocations_over_time.filter(
    d => d.elem === 'summarized' && d.opacity === 0.3);
  assertEqual(pool_summaries.length, 1, '1 per-pool summarized stripe');
  const ps_max = Math.max(...pool_summaries[0].size);
  assertEqual(ps_max, 3000, 'per-pool summarized max = 3000');

  // Global summarized band has ALL default pool allocs (100+200+...+1000=5500)
  const global_summary = result.allocations_over_time.find(
    d => d.elem === 'summarized' && d.opacity === undefined);
  assert(global_summary !== undefined, 'global summarized band exists');
  const gs_max = Math.max(...global_summary.size);
  assertEqual(gs_max, 5500, 'global summarized max = 5500 (all default pool allocs)');

  assertEqual(result.elements_length, 20, 'elements_length = 20 (total elements)');
}

function test_per_pool_summarization_with_frees() {
  console.log('test_per_pool_summarization_with_frees');
  // 6 allocs in private pool, limit to 3. Then free 2 drawn and 2 non-drawn.
  // Verify summarized stripe shrinks on non-drawn frees.
  const poolId = [1, 60];
  const snapshot = makeSnapshot({
    traces: [
      // 6 allocs: sizes 100,200,300,400,500,600
      // Top 3 drawn: 400,500,600. Summarized: 100,200,300.
      { action: 'alloc', addr: 0x8100, size: 100, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x8200, size: 200, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x8300, size: 300, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x8400, size: 400, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x8500, size: 500, frames: [], stream: 0 },
      { action: 'alloc', addr: 0x8600, size: 600, frames: [], stream: 0 },
      // Free a drawn element (600)
      { action: 'free_completed', addr: 0x8600, size: 600, frames: [], stream: 0 },
      // Free a non-drawn element (100) — summarized should shrink
      { action: 'free_completed', addr: 0x8100, size: 100, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0x8000, total_size: 0x1000, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 3, true);

  // After all events: drawn active = 400+500 = 900, summarized active = 200+300 = 500
  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  // 3 drawn stripes created (400, 500, 600), 600 was freed (closed out)
  assertEqual(stripes.length, 3, '3 pool stripes created (one freed)');

  const pool_summaries = result.allocations_over_time.filter(
    d => d.elem === 'summarized' && d.opacity === 0.3);
  assertEqual(pool_summaries.length, 1, '1 per-pool summarized stripe');

  // Summarized stripe should show the shrink: peak was 600 (100+200+300),
  // then 100 was freed → final = 500 (200+300)
  const ps = pool_summaries[0];
  const ps_max = Math.max(...ps.size);
  assertEqual(ps_max, 600, 'per-pool summarized peak = 600 (before non-drawn free)');
  const ps_final = ps.size.at(-1);
  assertEqual(ps_final, 500, 'per-pool summarized final = 500 (after non-drawn free)');
}

function test_per_pool_summarization_initially_allocated() {
  console.log('test_per_pool_summarization_initially_allocated');
  // 4 free_completed events in a private pool (no matching allocs).
  // Limit to 2. The 2 largest should be drawn stripes, the 2 smallest
  // should be in the per-pool summarized stripe.
  const poolId = [1, 70];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'free_completed', addr: 0xa100, size: 100, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0xa200, size: 200, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0xa300, size: 300, frames: [], stream: 0 },
      { action: 'free_completed', addr: 0xa400, size: 400, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0xa000, total_size: 0x1000, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 2, true);

  // Top 2 by size: 300, 400 → drawn stripes (pre-loaded then freed)
  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 2, '2 drawn stripes (largest initially_allocated)');
  const stripe_sizes = stripes.map(s => s.size).sort((a, b) => a - b);
  assertEqual(stripe_sizes[0], 300, 'drawn stripe 300');
  assertEqual(stripe_sizes[1], 400, 'drawn stripe 400');

  // Summarized: 100 + 200 = 300 initially, then freed to 0
  const pool_summaries = result.allocations_over_time.filter(
    d => d.elem === 'summarized' && d.opacity === 0.3);
  assertEqual(pool_summaries.length, 1, '1 per-pool summarized stripe');
  const ps_max = Math.max(...pool_summaries[0].size);
  assertEqual(ps_max, 300, 'per-pool summarized peak = 300 (100+200)');
  assertEqual(pool_summaries[0].size.at(-1), 0,
    'per-pool summarized final = 0 (all freed)');
}

function test_per_pool_summarization_interleaved() {
  console.log('test_per_pool_summarization_interleaved');
  // Drawn and non-drawn allocs interleaved: drawn stripes must not overlap
  // with the summarized region. Alloc order: 1800 (drawn), 200 (non-drawn),
  // 1600 (drawn), 100 (non-drawn).
  const poolId = [1, 80];
  const snapshot = makeSnapshot({
    traces: [
      { action: 'alloc', addr: 0xc000, size: 1800, frames: [], stream: 0 },
      { action: 'alloc', addr: 0xc800, size: 200, frames: [], stream: 0 },
      { action: 'alloc', addr: 0xd000, size: 1600, frames: [], stream: 0 },
      { action: 'alloc', addr: 0xd800, size: 100, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 0xc000, total_size: 0x2000, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  // max_entries=2: drawn = 1800, 1600. Non-drawn = 200, 100.
  const result = process_alloc_data(snapshot, 0, false, 2, true);

  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 2, '2 drawn stripes');

  const pool_summaries = result.allocations_over_time.filter(
    d => d.elem === 'summarized' && d.opacity === 0.3);
  assertEqual(pool_summaries.length, 1, '1 per-pool summarized stripe');

  // Summarized sits on top of drawn stripes (like global summarized band).
  // Drawn stripes start at envelope base, summarized is above them.
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  const env_base = envelopes[0].offsets.at(-1);
  const sum_final = pool_summaries[0].size.at(-1);
  assertEqual(sum_final, 300, 'summarized final = 300 (200+100)');

  // Drawn stripes should start at envelope base
  for (const stripe of stripes) {
    const final_offset = stripe.offsets.at(-1);
    assert(final_offset >= env_base,
      `stripe offset ${final_offset} must be >= env_base(${env_base})`);
  }

  // Summarized stripe offset should be at envelope_base + drawn_active
  const sum_offset = pool_summaries[0].offsets.at(-1);
  const drawn_tops = stripes.map(s => s.offsets.at(-1) + s.size);
  const max_drawn_top = Math.max(...drawn_tops);
  assert(sum_offset >= max_drawn_top - 1,
    `summarized offset ${sum_offset} should be at or above top of drawn stripes ${max_drawn_top}`);
}

// ============================================================
// Run all tests
// ============================================================

test_basic_alloc_free();
test_free_completed_is_matched();
test_pool_free_without_alloc_no_inflation();
test_pool_alloc_then_free_normal();
test_multiple_pool_frees_without_alloc();
test_non_pool_free_without_alloc();
test_mixed_pool_and_nonpool();
test_include_private_inactive_false_ignores_pools();
test_formatSize_bytes();
test_formatSize_kib();
test_formatSize_mib_gib();
test_formatSize_no_bytes();
test_formatAddr_block_event();
test_formatAddr_segment_event();
test_formatAddr_free_event();
test_elideRepeats_no_repeats();
test_elideRepeats_two_consecutive();
test_elideRepeats_three_or_more();
test_elideRepeats_mixed();
test_elideRepeats_empty();
test_context_for_id_with_pool();
test_context_for_id_unknown_pool();
test_context_for_id_free_without_alloc();
test_post177717_pool_id_from_trace_event();
test_post177717_pool_free_without_alloc_no_segment();
test_post177717_mixed_events_with_and_without_pool_id();
test_pool_grouped_by_stream();
test_segment_snapshot_with_trace_history();
test_segment_snapshot_no_trace();
test_default_pool_ghost_block();
test_ghost_blocks();
test_ghost_blocks_not_created_for_traced_addrs();
test_ghost_blocks_default_pool_collected();
test_ghost_blocks_not_in_segment_mode();
test_ghost_blocks_private_pool();
test_ghost_stripe_offset_with_multiple_pools();
test_full_snapshot_private_pools();
test_full_snapshot_no_private_pools();
test_envelope_grows_on_segment_map();
test_envelope_from_initial_reserved();
test_envelope_segment_map_no_double_count();
test_envelope_active_exceeds_reserved();
test_envelope_default_pool_unaffected();
test_per_pool_summarization();
test_per_pool_summarization_with_frees();
test_per_pool_summarization_initially_allocated();
test_per_pool_summarization_interleaved();

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
