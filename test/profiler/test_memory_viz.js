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
      // Note: real trace events never have segment_pool_id; pool is resolved via find_pool_id
      // from the segment address ranges.
      { action: 'free_completed', addr: 5000, size: 1000, frames: [], stream: 0 },
    ],
    segments: [{
      device: 0, address: 4096, total_size: 8192, segment_pool_id: poolId,
      stream: 0, blocks: [],
    }],
  });

  const result = process_alloc_data(snapshot, 0, false, 15000, true);
  // The free_completed creates an element in initially_allocated + actions.
  // The fix pre-loads it into the pool (pool.active=1000, envelope grows to 1000).
  // Then in actions, it's recognized as a free (pool.active goes to 0).
  // Peak should be 1000 (the initial state), NOT 2000.
  assert(result.max_size <= 1000,
    `pool free-without-alloc should not inflate peak: got ${result.max_size}`);

  // 1 element: the free_completed event that created an initially_allocated entry
  assertEqual(result.elements_length, 1, 'should have 1 element');

  // max_at_time should show the pool envelope at 1000, then dropping after the free
  assert(result.max_at_time.length > 0, 'max_at_time should not be empty');
  assertEqual(Math.max(...result.max_at_time), 1000,
    'max_at_time peak should be 1000 (envelope high-water mark)');

  // allocations_over_time should contain:
  //   - pool envelope (elem = "pool:1,42")
  //   - stripe for the block (elem = 0, the element index)
  //   - summarized_mem
  const envelopes = result.allocations_over_time.filter(
    d => typeof d.elem === 'string' && d.elem.startsWith('pool:'));
  assertEqual(envelopes.length, 1, 'should have 1 pool envelope');
  assertEqual(envelopes[0].elem, 'pool:1,42', 'envelope key matches pool id');

  const stripes = result.allocations_over_time.filter(
    d => typeof d.elem === 'number' && d.opacity === 0.5);
  assertEqual(stripes.length, 1, 'should have 1 pool stripe');
  assertEqual(stripes[0].size, 1000, 'stripe size matches block size');
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
  assertEqual(result.max_size, 500, 'normal pool alloc/free peak should be 500');
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
  // 3 blocks of 500 each were initially allocated. Pool envelope = 1500.
  // Then all 3 are freed. Peak should be 1500 (the initial state).
  // BUG would give 3000+ (each free treated as a new alloc).
  assert(result.max_size <= 1500,
    `multiple pool frees should not inflate: got ${result.max_size}, expected <= 1500`);
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
  // Peak: 200 (non-pool) + 400 (pool envelope) = 600
  assertEqual(result.max_size, 600, 'mixed pool+nonpool peak should be 600');
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

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
