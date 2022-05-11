#pragma once

// ${generated_comment}

#include <ATen/core/Tensor.h>
#include <ATen/core/IListRef.h>
#include <ATen/TensorMeta.h>
#include <ATen/TensorIterator.h>

${NativeMetaFunctions_includes}

namespace at {

namespace meta {

struct TORCH_API structured_mul_Tensor : public TensorIteratorBase {
    void meta(const at::Tensor & self, const at::Tensor & other);
};
${NativeMetaFunctions_declarations}

} // namespace meta
} // namespace at
