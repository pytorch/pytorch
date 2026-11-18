//! The chunks of the interpreter's stack of frames, kept when it frees them.
//!
//! The interpreter holds the frames of a thread in chunks of 16 KB, and gives
//! a chunk back to the system the moment the frame at its start returns. A
//! call whose frame is the first not to fit in the current chunk maps a chunk
//! and unmaps it again, and so does the next call from the same place. The
//! code of a compiled graph has one large frame and launches its kernels from
//! it: when that frame ends within a few hundred bytes of a chunk's end, each
//! launch pays for a pair of system calls, some 15 us. Whether it does
//! depends on the depth of the caller, to the byte, and not on the graph.
//!
//! So a chunk of that size that the interpreter frees is kept for its next
//! request of one. The interpreter asks its arena allocator for chunks, which
//! is one for the process and public (`PyObject_SetArenaAllocator`): what is
//! put in front of it here hands everything else through.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::OnceLock;

#[repr(C)]
#[derive(Clone, Copy)]
struct ArenaAllocator {
    ctx: *mut c_void,
    alloc: Option<unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void>,
    free: Option<unsafe extern "C" fn(*mut c_void, *mut c_void, usize)>,
}

// Set once, with the GIL held, and only read after.
unsafe impl Send for ArenaAllocator {}
unsafe impl Sync for ArenaAllocator {}

extern "C" {
    fn PyObject_GetArenaAllocator(allocator: *mut ArenaAllocator);
    fn PyObject_SetArenaAllocator(allocator: *mut ArenaAllocator);
}

/// The size of a chunk for frames of ordinary size (`DATA_STACK_CHUNK_SIZE`).
/// The arenas of the object allocator, the other thing asked of the arena
/// allocator, are 256 KB or 1 MB.
const CHUNK: usize = 16 * 1024;
/// A thread goes back and forth over one end of a chunk at a time.
const KEPT: usize = 4;

static BELOW: OnceLock<ArenaAllocator> = OnceLock::new();
static CHUNKS: [AtomicPtr<c_void>; KEPT] = [const { AtomicPtr::new(null_mut()) }; KEPT];

unsafe extern "C" fn alloc(_ctx: *mut c_void, size: usize) -> *mut c_void {
    if size == CHUNK {
        for slot in &CHUNKS {
            let kept = slot.swap(null_mut(), Ordering::AcqRel);
            if !kept.is_null() {
                return kept;
            }
        }
    }
    match BELOW.get().and_then(|below| below.alloc.map(|alloc| alloc(below.ctx, size))) {
        Some(made) => made,
        None => null_mut(),
    }
}

unsafe extern "C" fn free(_ctx: *mut c_void, ptr: *mut c_void, size: usize) {
    if size == CHUNK {
        for slot in &CHUNKS {
            if slot.compare_exchange(null_mut(), ptr, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
                return;
            }
        }
    }
    if let Some(below) = BELOW.get() {
        if let Some(free) = below.free {
            free(below.ctx, ptr, size);
        }
    }
}

/// Starts keeping the chunks the interpreter frees. Called with the GIL held,
/// which is what the interpreter calls its arena allocator under.
pub fn keep() {
    BELOW.get_or_init(|| unsafe {
        let mut below = ArenaAllocator { ctx: null_mut(), alloc: None, free: None };
        PyObject_GetArenaAllocator(&mut below);
        let mut mine = ArenaAllocator { ctx: null_mut(), alloc: Some(alloc), free: Some(free) };
        PyObject_SetArenaAllocator(&mut mine);
        below
    });
}
