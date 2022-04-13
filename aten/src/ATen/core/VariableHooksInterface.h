#pragma once

#include <c10/macros/Export.h>
#include <ATen/core/Tensor.h>

// A little explanation about why this file exists at all.  We have
// a few methods on Tensor class which require access to reified access to
// AutogradMeta.  In open source, this isn't a big deal: we just access
// torch/csrc/autograd/variable.h from aten/src/ATen/core/Tensor.cpp and
// we can put the definitions inline.  This is because everything gets balled
// into a single dynamic library in the end.
//
// However, inside our Facebook internal version of our build system, we
// have a split between aten and torch/csrc.  So we cannot simply just
// cross this boundary.  "Now wait," you might say, "Why don't we just
// merge the libraries inside Facebook".  Well, the problem is that there
// are some downstream applications which are at binary size limit, and
// incorporating all of the extra code from libtorch would push them
// over (admarket/adreview/service:adreviewservice, see also
// https://github.com/pytorch/pytorch/pull/29299)  So if you want to do that,
// we have to fix all of the services like this.
//
// I didn't want to block eliminating Tensor-Variable on this work, so I
// had to introduce another dynamic dispatch to get to the variable
// implementations (which live in torch/csrc/autograd/variable.cpp, FYI).
//
// I also considered using our existing dynamic dispatch mechanism, c10
// dispatcher, to do this.  However, (1) some of the functions on Tensor
// have weird signatures that are not supported by autograd, and (2)
// see this bug https://github.com/pytorch/pytorch/issues/30102

namespace torch { namespace autograd {

struct Node;

}} // namespace torch::autograd

namespace at {
namespace impl {

struct TORCH_API VariableHooksInterface {
  virtual ~VariableHooksInterface() = default;
  virtual TensorBase tensor_data(const TensorBase&) const = 0;
  virtual TensorBase variable_data(const TensorBase&) const = 0;
  virtual const std::shared_ptr<torch::autograd::Node>& grad_fn(const TensorBase&) const = 0;
  virtual unsigned _register_hook(
      const TensorBase&,
      std::function<TensorBase(const TensorBase&)> hook) const = 0;
  virtual void remove_hook(const TensorBase&, unsigned pos) const = 0;
  virtual bool is_view(const TensorBase&) const = 0;
  virtual const TensorBase& base(const TensorBase&) const = 0;
  virtual const std::string& name(const TensorBase&) const = 0;
  virtual bool is_leaf(const TensorBase&) const = 0;
  virtual int64_t output_nr(const TensorBase&) const = 0;
  virtual void set_data(const TensorBase&, const TensorBase&) const = 0;
  virtual TensorBase data(const TensorBase&) const = 0;
  virtual int64_t _version(const TensorBase&) const = 0;
  virtual void retain_grad(const TensorBase&) const = 0;
  virtual bool retains_grad(const TensorBase&) const = 0;
  virtual void _backward(const Tensor&, TensorList, const c10::optional<Tensor>&, c10::optional<bool>, bool) const = 0;
  virtual void requires_grad_(const TensorBase&, bool) const = 0;
};

TORCH_API void SetVariableHooks(VariableHooksInterface* hooks);
TORCH_API VariableHooksInterface* GetVariableHooks();

struct TORCH_API VariableHooksRegisterer {
  explicit VariableHooksRegisterer(VariableHooksInterface* hooks) {
    SetVariableHooks(hooks);
  }
};

}} // namespace at::impl
