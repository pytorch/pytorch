# C++ Docs Comparison: Published (pytorch/cppdocs) vs Local RST

**Context**: The published site at `docs.pytorch.org/cppdocs/api/` contains ~8,400 auto-generated Exhale/Doxygen pages (one per class, struct, namespace, function, file, etc.). Your local RST source uses hand-authored topic pages with `breathe` directives. This comparison checks whether significant public API classes/functions from the published Exhale pages are covered by your local RST files.

## Summary

- **Total significant classes/APIs in published docs**: ~170 (excluding `*Impl`, internal/detail, and low-level pages)
- **Covered in your local RST**: ~135
- **Missing (significant)**: ~35

---

## Missing Significant APIs (should consider adding)

### torch::nn (Modules)

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::nn::ReLU6` | Yes | **MISSING** | Medium | Activation module |
| `torch::nn::LogSigmoid` | Yes | **MISSING** | Medium | Activation module |
| `torch::nn::GLU` | Yes | **MISSING** | Medium | Activation module |
| `torch::nn::Softmax2d` | Yes | **MISSING** | Low | Rarely used |
| `torch::nn::MaxUnpool1d` | Yes | **MISSING** | Medium | Pooling module |
| `torch::nn::MaxUnpool2d` | Yes | **MISSING** | Medium | Pooling module |
| `torch::nn::MaxUnpool3d` | Yes | **MISSING** | Medium | Pooling module |
| `torch::nn::CosineSimilarity` | Yes | **MISSING** | Medium | Utility module |
| `torch::nn::PairwiseDistance` | Yes | **MISSING** | Medium | Utility module |
| `torch::nn::CrossMapLRN2d` | Yes | **MISSING** | Low | Legacy normalization |
| `torch::nn::AdaptiveLogSoftmaxWithLoss` | Yes | **MISSING** | Low | Specialized module |
| `torch::nn::Functional` | Yes | **MISSING** | Medium | Container module |
| `torch::nn::ModuleHolder` | Yes | **MISSING** | Medium | Core template wrapper |
| `torch::nn::AnyValue` | Yes | **MISSING** | Low | Used with AnyModule |
| `torch::nn::NamedAnyModule` | Yes | **MISSING** | Low | Used with Sequential |
| `torch::nn::PackedSequence` | Yes | **MISSING** | Medium | RNN utility |
| `torch::nn::TransformerEncoderLayer` | Yes | Partial (Impl only) | Low | Holder class not documented, Impl is |
| `torch::nn::TransformerDecoderLayer` | Yes | Partial (Impl only) | Low | Holder class not documented, Impl is |

### torch::optim (Optimizers)

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::optim::StepLR` | Yes | **MISSING** | High | Common scheduler |
| `torch::optim::ReduceLROnPlateauScheduler` | Yes | **MISSING** | High | Common scheduler |
| `torch::optim::OptimizerOptions` | Yes | **MISSING** | Medium | Base class for all optimizer options |
| `torch::optim::OptimizerParamGroup` | Yes | **MISSING** | Medium | Core optimizer API |
| `torch::optim::OptimizerParamState` | Yes | **MISSING** | Medium | Core optimizer API |

### torch::data (Data Loading)

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::data::Iterator` | Yes | **MISSING** | Medium | DataLoader iteration |
| `torch::data::StatefulDataLoader` | Yes | **MISSING** | Medium | DataLoader variant |
| `torch::data::StatelessDataLoader` | Yes | **MISSING** | Medium | DataLoader variant |
| `torch::data::datasets::ChunkDataReader` | Yes | **MISSING** | Low | Specialized dataset |
| `torch::data::datasets::ChunkDataset` | Yes | **MISSING** | Low | Specialized dataset |
| `torch::data::datasets::MapDataset` | Yes | **MISSING** | Medium | Core dataset type |
| `torch::data::datasets::SharedBatchDataset` | Yes | **MISSING** | Low | Specialized dataset |
| `torch::data::datasets::StatefulDataset` | Yes | **MISSING** | Low | Dataset base class |
| `torch::data::samplers::DistributedSampler` | Yes | **MISSING** | Medium | Base for distributed samplers |
| `torch::data::samplers::DistributedSequentialSampler` | Yes | **MISSING** | Medium | Distributed training |
| `torch::data::samplers::StreamSampler` | Yes | **MISSING** | Low | Streaming data |
| `torch::data::transforms::BatchLambda` | Yes | **MISSING** | Low | Transform utility |
| `torch::data::transforms::BatchTransform` | Yes | **MISSING** | Low | Transform base |
| `torch::data::transforms::Lambda` | Yes | **MISSING** | Low | Transform utility |
| `torch::data::transforms::TensorLambda` | Yes | **MISSING** | Low | Transform utility |
| `torch::data::transforms::TensorTransform` | Yes | **MISSING** | Low | Transform utility |
| `torch::data::transforms::Transform` | Yes | **MISSING** | Medium | Transform base class |

### torch (Top-level)

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::CppFunction` | Yes | **MISSING** | Medium | Operator registration |
| `torch::OrderedDict` | Yes | **MISSING** | Medium | Used by nn::Module |
| `torch::ExpandingArray` | Yes | **MISSING** | Low | Internal utility |
| `torch::ExpandingArrayWithOptionalElem` | Yes | **MISSING** | Low | Internal utility |
| `torch::IMethod` | Yes | **MISSING** | Low | JIT interface |
| `torch::CustomClassHolder` | Yes | **MISSING** | Low | Custom class base |

### torch::autograd

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::autograd::NodeGuard` | Yes | **MISSING** | Low | Internal autograd |

### at (ATen)

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `at::Tensor` | Yes | **MISSING** | High | Core tensor class (documented in aten/tensor.rst via prose, but no doxygenclass directive) |
| `at::TensorRef` | Yes | **MISSING** | Low | Internal reference type |
| `at::OptionalTensorRef` | Yes | **MISSING** | Low | Internal reference type |
| `at::native::*Descriptor` | Yes | **MISSING** | Low | cuDNN/MKL descriptors (5 classes) |

### c10 (Core)

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `c10::Dict` | Yes | **MISSING** | Medium | Container type |
| `c10::List` | Yes | **MISSING** | Medium | Container type |
| `c10::IListRef` | Yes | **MISSING** | Low | Internal list ref |
| `c10::Error` + subclasses | Yes | **MISSING** | Low | 12 error/exception classes |
| `c10::Warning` | Yes | **MISSING** | Low | Warning infrastructure |

### torch::stable

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::stable::accelerator::Stream` | Yes | **MISSING** | Medium | Stable ABI stream |

### torch::nativert

| Class | Published? | In Local RST? | Priority | Notes |
|-------|-----------|---------------|----------|-------|
| `torch::nativert::ModelRunnerHandle` | Yes | **MISSING** | Low | NativERT runtime (new) |

---

## Fully Covered Sections (no gaps)

- **torch::nn**: ~95% covered — only missing a handful of less common modules
- **torch::serialize**: Fully covered (InputArchive, OutputArchive, save/load functions)
- **torch::stable**: Mostly covered (missing Stream only)
- **CUDA guards/streams/utilities**: Fully covered
- **XPU streams/utilities**: Fully covered
- **c10 guards/streams/types**: Covered (Device, DeviceGuard, OptionalDeviceGuard, Stream, ArrayRef, OptionalArrayRef)

## Top Priority Gaps

1. **`torch::optim::StepLR` and `ReduceLROnPlateauScheduler`** — Common schedulers, documented in published docs but missing locally
2. **`at::Tensor`** — The central class; your `aten/tensor.rst` covers it in prose but has no `doxygenclass` directive
3. **`torch::nn::MaxUnpool{1,2,3}d`** — Standard pooling counterparts, present in published docs
4. **`torch::nn::ReLU6`, `LogSigmoid`, `GLU`** — Common activation modules
5. **`c10::Dict`, `c10::List`** — Core container types used across the API
6. **`torch::data::transforms::Transform`** — Base class for all transforms
7. **`torch::stable::accelerator::Stream`** — Part of stable ABI (new)
