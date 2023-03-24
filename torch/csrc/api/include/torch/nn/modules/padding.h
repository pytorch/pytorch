#pragma once

#include <torch/expanding_array.h>
#include <torch/nn/cloneable.h>
#include <torch/nn/functional/padding.h>

#include <torch/csrc/Export.h>

namespace torch {
namespace nn {

/// Base class for all (dimension-specialized) ReflectionPad modules.
template <size_t D, typename Derived>
class TORCH_API ReflectionPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ReflectionPadImpl(ExpandingArray<D * 2> padding)
      : ReflectionPadImpl(ReflectionPadOptions<D>(padding)) {}
  explicit ReflectionPadImpl(const ReflectionPadOptions<D>& options_);

  void reset() override;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `ReflectionPad{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReflectionPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReflectionPad1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReflectionPad1d model(ReflectionPad1dOptions({3, 1}));
/// ```
class TORCH_API ReflectionPad1dImpl
    : public ReflectionPadImpl<1, ReflectionPad1dImpl> {
 public:
  using ReflectionPadImpl<1, ReflectionPad1dImpl>::ReflectionPadImpl;
};

/// A `ModuleHolder` subclass for `ReflectionPad1dImpl`.
/// See the documentation for `ReflectionPad1dImpl` class to learn what methods
/// it provides, and examples of how to use `ReflectionPad1d` with
/// `torch::nn::ReflectionPad1dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReflectionPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReflectionPad2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReflectionPad2d model(ReflectionPad2dOptions({1, 1, 2, 0}));
/// ```
class TORCH_API ReflectionPad2dImpl
    : public ReflectionPadImpl<2, ReflectionPad2dImpl> {
 public:
  using ReflectionPadImpl<2, ReflectionPad2dImpl>::ReflectionPadImpl;
};

/// A `ModuleHolder` subclass for `ReflectionPad2dImpl`.
/// See the documentation for `ReflectionPad2dImpl` class to learn what methods
/// it provides, and examples of how to use `ReflectionPad2d` with
/// `torch::nn::ReflectionPad2dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReflectionPad2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReflectionPad3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReflectionPad over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReflectionPad3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReflectionPad3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReflectionPad3d model(ReflectionPad3dOptions(1));
/// ReflectionPad3d model(ReflectionPad3dOptions({1, 1, 2, 0, 1, 2}));
/// ```
class TORCH_API ReflectionPad3dImpl
    : public ReflectionPadImpl<3, ReflectionPad3dImpl> {
 public:
  using ReflectionPadImpl<3, ReflectionPad3dImpl>::ReflectionPadImpl;
};

/// A `ModuleHolder` subclass for `ReflectionPad3dImpl`.
/// See the documentation for `ReflectionPad3dImpl` class to learn what methods
/// it provides, and examples of how to use `ReflectionPad3d` with
/// `torch::nn::ReflectionPad3dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReflectionPad3d);

// ============================================================================

/// Base class for all (dimension-specialized) ReplicationPad modules.
template <size_t D, typename Derived>
class TORCH_API ReplicationPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ReplicationPadImpl(ExpandingArray<D * 2> padding)
      : ReplicationPadImpl(ReplicationPadOptions<D>(padding)) {}
  explicit ReplicationPadImpl(const ReplicationPadOptions<D>& options_);

  void reset() override;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `ReplicationPad{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ReplicationPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReplicationPad1d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReplicationPad over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReplicationPad1d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReplicationPad1dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReplicationPad1d model(ReplicationPad1dOptions({3, 1}));
/// ```
class TORCH_API ReplicationPad1dImpl
    : public ReplicationPadImpl<1, ReplicationPad1dImpl> {
 public:
  using ReplicationPadImpl<1, ReplicationPad1dImpl>::ReplicationPadImpl;
};

/// A `ModuleHolder` subclass for `ReplicationPad1dImpl`.
/// See the documentation for `ReplicationPad1dImpl` class to learn what methods
/// it provides, and examples of how to use `ReplicationPad1d` with
/// `torch::nn::ReplicationPad1dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReplicationPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReplicationPad2d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReplicationPad over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReplicationPad2d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReplicationPad2dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReplicationPad2d model(ReplicationPad2dOptions({1, 1, 2, 0}));
/// ```
class TORCH_API ReplicationPad2dImpl
    : public ReplicationPadImpl<2, ReplicationPad2dImpl> {
 public:
  using ReplicationPadImpl<2, ReplicationPad2dImpl>::ReplicationPadImpl;
};

/// A `ModuleHolder` subclass for `ReplicationPad2dImpl`.
/// See the documentation for `ReplicationPad2dImpl` class to learn what methods
/// it provides, and examples of how to use `ReplicationPad2d` with
/// `torch::nn::ReplicationPad2dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReplicationPad2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ReplicationPad3d
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ReplicationPad over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ReplicationPad3d to
/// learn about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ReplicationPad3dOptions` class to
/// learn what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ReplicationPad3d model(ReplicationPad3dOptions({1, 2, 1, 2, 1, 2}));
/// ```
class TORCH_API ReplicationPad3dImpl
    : public ReplicationPadImpl<3, ReplicationPad3dImpl> {
 public:
  using ReplicationPadImpl<3, ReplicationPad3dImpl>::ReplicationPadImpl;
};

/// A `ModuleHolder` subclass for `ReplicationPad3dImpl`.
/// See the documentation for `ReplicationPad3dImpl` class to learn what methods
/// it provides, and examples of how to use `ReplicationPad3d` with
/// `torch::nn::ReplicationPad3dOptions`. See the documentation for
/// `ModuleHolder` to learn about PyTorch's module storage semantics.
TORCH_MODULE(ReplicationPad3d);

// ============================================================================

/// Base class for all (dimension-specialized) ZeroPad modules.
template <size_t D, typename Derived>
class TORCH_API ZeroPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ZeroPadImpl(ExpandingArray<D * 2> padding)
      : ZeroPadImpl(ZeroPadOptions<D>(padding)) {}
  explicit ZeroPadImpl(const ZeroPadOptions<D>& options_);

  void reset() override;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `ZeroPad{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ZeroPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ZeroPad1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Applies ZeroPad over a 1-D input.
class TORCH_API ZeroPad1dImpl : public ZeroPadImpl<1, ZeroPad1dImpl> {
 public:
  using ZeroPadImpl<1, ZeroPad1dImpl>::ZeroPadImpl;
};

/// A `ModuleHolder` subclass for `ZeroPad1dImpl`.
/// See the documentation for `ZeroPad1dImpl` class to learn what methods it
/// provides, and examples of how to use `ZeroPad1d` with
/// `torch::nn::ZeroPad1dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(ZeroPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ZeroPad2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Applies ZeroPad over a 2-D input.
class TORCH_API ZeroPad2dImpl : public ZeroPadImpl<2, ZeroPad2dImpl> {
 public:
  using ZeroPadImpl<2, ZeroPad2dImpl>::ZeroPadImpl;
};

/// A `ModuleHolder` subclass for `ZeroPad2dImpl`.
/// See the documentation for `ZeroPad2dImpl` class to learn what methods it
/// provides, and examples of how to use `ZeroPad2d` with
/// `torch::nn::ZeroPad2dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(ZeroPad2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ZeroPad3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Applies ZeroPad over a 3-D input.
class TORCH_API ZeroPad3dImpl : public ZeroPadImpl<3, ZeroPad3dImpl> {
 public:
  using ZeroPadImpl<3, ZeroPad3dImpl>::ZeroPadImpl;
};

/// A `ModuleHolder` subclass for `ZeroPad3dImpl`.
/// See the documentation for `ZeroPad3dImpl` class to learn what methods it
/// provides, and examples of how to use `ZeroPad3d` with
/// `torch::nn::ZeroPad3dOptions`. See the documentation for `ModuleHolder` to
/// learn about PyTorch's module storage semantics.
TORCH_MODULE(ZeroPad3d);

// ============================================================================

/// Base class for all (dimension-specialized) ConstantPad modules.
template <size_t D, typename Derived>
class TORCH_API ConstantPadImpl : public torch::nn::Cloneable<Derived> {
 public:
  ConstantPadImpl(ExpandingArray<D * 2> padding, double value)
      : ConstantPadImpl(ConstantPadOptions<D>(padding, value)) {}
  explicit ConstantPadImpl(const ConstantPadOptions<D>& options_);

  void reset() override;

  Tensor forward(const Tensor& input);

  /// Pretty prints the `ConstantPad{1,2}d` module into the given `stream`.
  void pretty_print(std::ostream& stream) const override;

  /// The options with which this `Module` was constructed.
  ConstantPadOptions<D> options;
};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConstantPad1d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ConstantPad over a 1-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConstantPad1d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConstantPad1dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConstantPad1d model(ConstantPad1dOptions({3, 1}, 3.5));
/// ```
class TORCH_API ConstantPad1dImpl
    : public ConstantPadImpl<1, ConstantPad1dImpl> {
 public:
  using ConstantPadImpl<1, ConstantPad1dImpl>::ConstantPadImpl;
};

/// A `ModuleHolder` subclass for `ConstantPad1dImpl`.
/// See the documentation for `ConstantPad1dImpl` class to learn what methods it
/// provides, and examples of how to use `ConstantPad1d` with
/// `torch::nn::ConstantPad1dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConstantPad1d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConstantPad2d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ConstantPad over a 2-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConstantPad2d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConstantPad2dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConstantPad2d model(ConstantPad2dOptions({3, 0, 2, 1}, 3.5));
/// ```
class TORCH_API ConstantPad2dImpl
    : public ConstantPadImpl<2, ConstantPad2dImpl> {
 public:
  using ConstantPadImpl<2, ConstantPad2dImpl>::ConstantPadImpl;
};

/// A `ModuleHolder` subclass for `ConstantPad2dImpl`.
/// See the documentation for `ConstantPad2dImpl` class to learn what methods it
/// provides, and examples of how to use `ConstantPad2d` with
/// `torch::nn::ConstantPad2dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConstantPad2d);

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ConstantPad3d ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Applies ConstantPad over a 3-D input.
/// See https://pytorch.org/docs/master/nn.html#torch.nn.ConstantPad3d to learn
/// about the exact behavior of this module.
///
/// See the documentation for `torch::nn::ConstantPad3dOptions` class to learn
/// what constructor arguments are supported for this module.
///
/// Example:
/// ```
/// ConstantPad3d model(ConstantPad3dOptions({1, 2, 1, 2, 1, 2}, 3.5));
/// ```
class TORCH_API ConstantPad3dImpl
    : public ConstantPadImpl<3, ConstantPad3dImpl> {
 public:
  using ConstantPadImpl<3, ConstantPad3dImpl>::ConstantPadImpl;
};

/// A `ModuleHolder` subclass for `ConstantPad3dImpl`.
/// See the documentation for `ConstantPad3dImpl` class to learn what methods it
/// provides, and examples of how to use `ConstantPad3d` with
/// `torch::nn::ConstantPad3dOptions`. See the documentation for `ModuleHolder`
/// to learn about PyTorch's module storage semantics.
TORCH_MODULE(ConstantPad3d);

} // namespace nn
} // namespace torch
