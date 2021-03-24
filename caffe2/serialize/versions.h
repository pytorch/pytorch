#pragma once

#include <cstdint>

namespace caffe2 {
namespace serialize {

constexpr uint64_t kMinSupportedFileFormatVersion = 0x1L;
constexpr uint64_t kMaxSupportedFileFormatVersion = 0x6L;

// Versions (i.e. why was the version number bumped?)

// Note [Dynamic Versions and torch.jit.save vs. torch.save]
//
// Our versioning scheme has a "produced file format version" which
// describes how an archive is to be read. The version written in an archive
// is at least this current produced file format version, but may be greater
// if it includes certain symbols. We refer to these conditional versions
// as "dynamic," since they are identified at runtime.
//
// Dynamic versioning is useful when an operator's semantics are updated.
// When using torch.jit.save we want those semantics to be preserved. If
// we bumped the produced file format version on every change, however,
// then older versions of PyTorch couldn't read even simple archives, like
// a single tensor, from newer versions of PyTorch. Instead, we
// assign dynamic versions to these changes that override the
// produced file format version as needed. That is, when the semantics
// of torch.div changed it was assigned dynamic version 4, and when
// torch.jit.saving modules that use torch.div those archives also have
// (at least) version 4. This prevents earlier versions of PyTorch
// from accidentally performing the wrong kind of division. Modules
// that don't use torch.div or other operators with dynamic versions
// can write the produced file format version, and these programs will
// run as expected on earlier versions of PyTorch.
//
// While torch.jit.save attempts to preserve operator semantics,
// torch.save does not. torch.save is analogous to pickling Python, so
// a function that uses torch.div will have different behavior if torch.saved
// and torch.loaded across PyTorch versions. From a technical perspective,
// torch.save ignores dynamic versioning.

// 1. Initial version
// 2. Removed op_version_set version numbers
// 3. Added type tags to pickle serialization of container types
// 4. (Dynamic) Stopped integer division using torch.div
//      (a versioned symbol preserves the historic behavior of versions 1--3)
// 5. (Dynamic) Stops torch.full inferring a floating point dtype
//      when given bool or integer fill values.
// 6. Write version string to `./data/version` instead of `version`.
constexpr uint64_t kProducedFileFormatVersion = 0x3L;

// The version we write when the archive contains bytecode.
// It must be higher or eq to kProducedFileFormatVersion.
// Because torchscript changes is likely introduce bytecode change.
// If kProducedFileFormatVersion is increased, kProducedBytecodeVersion
// should be increased too. The relationship is:
// kMaxSupportedFileFormatVersion >= (most likely ==) kProducedBytecodeVersion
//   >= kProducedFileFormatVersion
// If a format change is forward compatible (still readable by older
// executables), we will not increment the version number, to minimize the
// risk of breaking existing clients. TODO: A better way would be to allow
// the caller that creates a model to specify a maximum version that its
// clients can accept.
// Versions:
//  0x1L: Initial version
//  0x2L: (Comment missing)
//  0x3L: (Comment missing)
//  0x4L: (Comment missing)
//  0x4L: (update) Added schema to function tuple. Forward-compatible change.
constexpr uint64_t kProducedBytecodeVersion = 0x4L;

static_assert(kProducedBytecodeVersion >= kProducedFileFormatVersion,
    "kProducedBytecodeVersion must be higher or equal to kProducedFileFormatVersion.");

// Introduce kMinSupportedBytecodeVersion for limited backward compatibility
// support of bytecode. If
// kMinSupportedBytecodeVersion <= model_version <= kProducedBytecodeVersion (in loader),
// we should support this model_version. For example, we provide a wrapper to
// handle an updated operator.
constexpr uint64_t kMinSupportedBytecodeVersion = 0x3L;
} // namespace serialize
} // namespace caffe2
