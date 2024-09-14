#pragma once
#include <cstdint>

namespace caffe2::serialize {

constexpr uint64_t kMinSupportedFileFormatVersion = 0x1L;

constexpr uint64_t kMaxSupportedFileFormatVersion = 0xAL;

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

// [12/15/2021]
// kProducedFileFormatVersion is set to 7 from 3 due to a different
// interpretation of what file format version is.
// Whenever there is new upgrader introduced,
// this number should be bumped.
// The reasons that version is bumped in the past:
//     1. aten::div is changed at version 4
//     2. aten::full is changed at version 5
//     3. torch.package uses version 6
//     4. Introduce new upgrader design and set the version number to 7
//        mark this change
// --------------------------------------------------
// We describe new operator version bump reasons here:
// 1) [01/24/2022]
//     We bump the version number to 8 to update aten::linspace
//     and aten::linspace.out to error out when steps is not
//     provided. (see: https://github.com/pytorch/pytorch/issues/55951)
// 2) [01/30/2022]
//     Bump the version number to 9 to update aten::logspace and
//     and aten::logspace.out to error out when steps is not
//     provided. (see: https://github.com/pytorch/pytorch/issues/55951)
// 3) [02/11/2022]
//     Bump the version number to 10 to update aten::gelu and
//     and aten::gelu.out to support the new approximate kwarg.
//     (see: https://github.com/pytorch/pytorch/pull/61439)
constexpr uint64_t kProducedFileFormatVersion = 0xAL;

// Absolute minimum version we will write packages. This
// means that every package from now on will always be
// greater than this number.
constexpr uint64_t kMinProducedFileFormatVersion = 0x3L;

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
//  0x4L: (update) Added schema to function tuple. Forward-compatible change.
//  0x5L: (update) Update bytecode is sharing constant tensor files from
//  torchscript, and only serialize extra tensors that are not in the
//  torchscript constant table. Also update tensor storage schema adapting to
//  the unify format, the root key of tensor storage is updated from {index} to
//  {the_pointer_value_the_tensor.storage}, for example:
//  `140245072983168.storage` Forward-compatibility change.
//  0x6L: Implicit opereator versioning using number of specified argument.
//  Refer to the summary of https://github.com/pytorch/pytorch/pull/56845 for
//  details.
//  0x7L: Enable support for operators with default arguments plus out
//  arguments. Refer. See https://github.com/pytorch/pytorch/pull/63651 for
//  details.
//  0x8L: Emit promoted operators as instructions. See
//  https://github.com/pytorch/pytorch/pull/71662 for details.
//  0x9L: Change serialization format from pickle to format This version is to
//  serve migration. v8 pickle and v9 flatbuffer are the same. Refer to the
//  summary of https://github.com/pytorch/pytorch/pull/75201 for more details.
constexpr uint64_t kProducedBytecodeVersion = 0x8L;

// static_assert(
//     kProducedBytecodeVersion >= kProducedFileFormatVersion,
//     "kProducedBytecodeVersion must be higher or equal to
//     kProducedFileFormatVersion.");

// Introduce kMinSupportedBytecodeVersion and kMaxSupportedBytecodeVersion
// for limited backward/forward compatibility support of bytecode. If
// kMinSupportedBytecodeVersion <= model_version <= kMaxSupportedBytecodeVersion
// (in loader), we should support this model_version. For example, we provide a
// wrapper to handle an updated operator.
constexpr uint64_t kMinSupportedBytecodeVersion = 0x4L;
constexpr uint64_t kMaxSupportedBytecodeVersion = 0x9L;

} // namespace caffe2::serialize
