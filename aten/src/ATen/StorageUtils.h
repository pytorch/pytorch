#pragma once

#include <c10/core/Storage.h>
#include <c10/core/StorageImpl.h>
#include <c10/util/intrusive_ptr.h>

namespace at {

class TensorBase;

// Here we define a series of utils to create/manipulate ATen backed
// c10 storage implementations.

/**
 * Create a new shared memory storage impl managed by file descriptor
 *
 * @param size  size in bytes
 */
C10_EXPORT c10::intrusive_ptr<c10::StorageImpl> new_shm_fd_storage(size_t size);

/**
 * Copy src to dst
 * Caller must guarantee the validness of the storage objects
 * during the entire copy process, esp. when it's async.
 *
 * This can probably live in c10 namespace later if needed,
 * but for now keep it in at to keep implementation simple.
 *
 * @param dst  dst tensor
 * @param src  src tensor
 * @param non_blocking  (default false) whether this operation blocks caller
 */
C10_EXPORT void storage_copy(
    c10::Storage& dst,
    const c10::Storage& src,
    bool non_blocking = false);

/**
 * In place change the storage to shm based.
 *
 * This would later be invoked by at::TensorBase user facing API.
 * For now, to keep the change minimal,
 * intentionally separate the API changes from the core logic,
 * as the API changes may also need to handle device/OS specifics.
 *
 * @param t  a tensor
 */
C10_EXPORT void share_memory_(TensorBase& t);

} // namespace at
