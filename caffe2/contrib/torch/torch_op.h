#pragma once
#include <unordered_map>

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

extern "C" {
#include <TH/THStorage.h>
#include <TH/THTensor.h>
#include <lua.h>
#include <luaT.h>
#include <lualib.h>
}

namespace caffe2 {

namespace torch {

template <typename Context>
struct TyTraits {};

template <>
struct TyTraits<CPUContext> {
  static const char* moduleTy;
  static const char* prelude;
  static const char* tensorTy;
  using Tensor = THFloatTensor;
};

template <typename Context>
class Torch final {
 public:
  using Traits = TyTraits<Context>;
  Torch() {
    L_ = luaL_newstate();
    luaL_openlibs(L_);
    luaL_loadstring(L_, Traits::prelude);
    int err = lua_pcall(L_, 0, 0, 0);
    CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L_, -1));
  };

  ~Torch() {
    lua_close(L_);
  }

  lua_State* L() {
    return L_;
  }

  static const char* tensorTy(const Blob& blob) {
    CAFFE_ENFORCE(blob.template IsType<Tensor<Context>>());
    const auto& tc = blob.template Get<Tensor<Context>>();
    CAFFE_ENFORCE(
        tc.template IsType<float>() + tc.meta().name(), ", ", tc.size());
    return Traits::tensorTy;
  }

  void setContext(Context* /*context*/) {}

  void setTensor(typename Traits::Tensor* t, Blob* blob) {
    CAFFE_ENFORCE_EQ(tensorTy(*blob), Traits::tensorTy);
    auto* tc = blob->template GetMutable<Tensor<Context>>();
    CAFFE_ENFORCE_EQ(THFloatTensor_nElement(t), tc->size());
    THFloatStorage* storage = THFloatStorage_newWithData(
        tc->template mutable_data<float>(), tc->size());
    THFloatStorage_clearFlag(storage, TH_STORAGE_FREEMEM);
    THFloatStorage* original = t->storage;
    t->storage = storage;
    THFloatStorage_free(original);
  }

  std::vector<TIndex> tensorShape(typename Traits::Tensor* t) {
    auto* size = t->size;
    return std::vector<TIndex>(size, size + THFloatTensor_nDimension(t));
  }

  typename Traits::Tensor* newTensorAs(const Tensor<Context>& tc) {
    THLongStorage* thshape = THLongStorage_newWithSize(tc.ndim());
    for (uint32_t i = 0; i < tc.ndim(); ++i) {
      THLongStorage_set(thshape, i, tc.dim(i));
    }
    THFloatTensor* d = THFloatTensor_newWithSize(thshape, nullptr);
    THLongStorage_free(thshape);
    return d;
  }

  typename Traits::Tensor* blobToTensor(Blob* blob) {
    CAFFE_ENFORCE_EQ(tensorTy(*blob), Traits::tensorTy);
    auto* tc = blob->template GetMutable<Tensor<Context>>();

    size_t size = tc->size();
    THLongStorage* thshape = THLongStorage_newWithSize(tc->ndim());
    for (int i = 0; i < tc->ndim(); ++i) {
      THLongStorage_set(thshape, i, tc->dim(i));
    }
    THFloatStorage* storage =
        THFloatStorage_newWithData(tc->template mutable_data<float>(), size);
    THFloatStorage_clearFlag(storage, TH_STORAGE_FREEMEM);
    auto* th = THFloatTensor_newWithStorage(storage, 0, thshape, nullptr);
    THFloatStorage_free(storage);
    THLongStorage_free(thshape);
    CAFFE_ENFORCE_EQ(
        THFloatTensor_storage(th)->data, tc->template mutable_data<float>());
    return th;
  }

  std::vector<typename Traits::Tensor*> pushTable(
      const std::vector<Blob*>& blobs) {
    if (blobs.empty()) {
      lua_pushnil(L());
      return {};
    }

    if (blobs.size() == 1) {
      auto* th = blobToTensor(blobs[0]);
      luaT_pushudata(L(), th, tensorTy(*blobs[0]));
      return {th};
    }

    std::vector<typename Traits::Tensor*> res;
    lua_createtable(L(), blobs.size(), 0);
    int index = 1;
    for (auto* blob : blobs) {
      auto* th = blobToTensor(blob);
      res.push_back(th);
      luaT_pushudata(L(), th, tensorTy(*blob));
      lua_rawseti(L(), -2, index++);
    }
    return res;
  }

  void verifyOutput(Blob* dst, typename Traits::Tensor* torchDst) {
    if (!luaT_isudata(L(), -1, Traits::tensorTy)) {
      LOG(FATAL) << "Unsupported Torch tensor type " << luaT_typename(L(), -1);
    }

    // Invariant: dst has the same size as src, and has the same data
    // values as src.
    auto* src = static_cast<typename Traits::Tensor*>(
        luaT_toudata(L(), -1, Traits::tensorTy));
    auto* thDst = static_cast<typename Traits::Tensor*>(torchDst);
    auto* tcDst = dst->template GetMutable<Tensor<Context>>();
    CAFFE_ENFORCE(src->storage->data);
    CAFFE_ENFORCE(src->storage->size);
    CAFFE_ENFORCE_EQ(src->storage->data, thDst->storage->data);
    CAFFE_ENFORCE_EQ(src->storage->data, tcDst->template data<float>());
    CAFFE_ENFORCE_EQ(src->storage->size, thDst->storage->size);
    CAFFE_ENFORCE_EQ(src->storage->size, tcDst->size());
  }

  void verifyOutputs(
      const std::vector<Blob*>& blobs,
      const std::vector<typename Traits::Tensor*>& tensors) {
    CAFFE_ENFORCE_EQ(tensors.size(), blobs.size());

    if (blobs.empty()) {
      return;
    }

    if (blobs.size() == 1) {
      verifyOutput(blobs[0], tensors[0]);
      return;
    }

    CAFFE_ENFORCE(lua_istable(L(), -1));
    lua_pushnil(L());
    for (auto i = 0; i < blobs.size(); ++i) {
      CAFFE_ENFORCE(lua_next(L(), -2));
      verifyOutput(blobs[i], tensors[i]);
      lua_pop(L(), 1);
    }
    lua_pop(L(), 1);
  }

private:
  lua_State* L_;
};
}

template <typename Context>
class TorchOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using OperatorBase::Outputs;
  using OperatorBase::Inputs;
  TorchOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {
    lua_State* L = state_.L();
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);
    const auto initString = "return " +
        OperatorBase::GetSingleArgument<std::string>("init", "") + ":" +
        torch::Torch<Context>::Traits::moduleTy + "()";
    CAFFE_ENFORCE_EQ(luaL_loadstring(L, initString.c_str()), 0);
    int err = lua_pcall(L, 0, 1, 0);
    CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L, -1));
    // Get number of parameters
    uint32_t numParams = 0;
    lua_getfield(L, -1, "parameters");
    lua_pushvalue(L, -2);
    CAFFE_ENFORCE_EQ(lua_pcall(L, 1, LUA_MULTRET, 0), 0);
    if (lua_gettop(L) == 1) {
      numParams = 0;
    } else {
      CAFFE_ENFORCE_EQ(lua_gettop(L), 3);
      numParams = lua_objlen(L, -2);
      lua_pop(L, 2);
    }
    CAFFE_ENFORCE_EQ(
        numParams, OperatorBase::GetSingleArgument<int>("num_params", 0));
    // TODO: free parameters?
    self_ = luaL_ref(L, LUA_REGISTRYINDEX);
  }

  void reshapeBlobs(
      const std::vector<Blob*>& inputBlobs,
      const std::vector<Blob*>& paramBlobs,
      const std::vector<Blob*>& outputBlobs) {
    auto cacheEqual = [=]() {
      if (cachedInputSizes_.size() != inputBlobs.size()) {
        return false;
      }

      for (auto i = 0; i < inputBlobs.size(); ++i) {
        const auto& current =
            inputBlobs[i]->template Get<Tensor<Context>>().dims();
        const auto& cached = cachedInputSizes_[i];
        if (current != cached) {
          return false;
        }
      }
      return true;
    };

    if (cacheEqual()) {
      return;
    }
    LOG(INFO) << "Cached blobs not equal, running :updateOutput to reshape";
    lua_State* L = state_.L();
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);
    lua_rawgeti(L, LUA_REGISTRYINDEX, self_);
    lua_getfield(L, -1, "updateOutput");
    lua_pushvalue(L, -2); // self
    if (inputBlobs.size() == 1) {
      const auto& tc = inputBlobs[0]->template Get<Tensor<Context>>();
      auto* inputData = state_.newTensorAs(tc);
      luaT_pushudata(L, inputData, torch::Torch<Context>::Traits::tensorTy);
    } else if (inputBlobs.size() > 1) {
      lua_createtable(L, inputBlobs.size(), 0);
      for (auto i = 0; i < inputBlobs.size(); ++i) {
        const auto* blob = inputBlobs[i];
        const auto& tc = blob->template Get<Tensor<Context>>();
        auto* inputData = state_.newTensorAs(tc);
        luaT_pushudata(L, inputData, torch::Torch<Context>::Traits::tensorTy);
        lua_rawseti(L, -2, i + 1);
      }
    }
    int err = lua_pcall(L, 2, 0, 0);
    CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L, -1));
    if (paramBlobs.size() != 0) {
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err2 = lua_pcall(L, 1, LUA_MULTRET, 0);
      CAFFE_ENFORCE_EQ(err2, 0);
      CAFFE_ENFORCE_EQ(lua_gettop(L), 3);
      lua_pushnil(L);
      int i = 0;
      while (lua_next(L, -3) && i < paramBlobs.size()) {
        CAFFE_ENFORCE(
            luaT_isudata(L, -1, torch::Torch<Context>::Traits::tensorTy));
        auto* param =
            static_cast<typename torch::Torch<Context>::Traits::Tensor*>(
                luaT_toudata(L, -1, torch::Torch<Context>::Traits::tensorTy));
        auto paramShape = state_.tensorShape(param);
        auto* blob = paramBlobs[i];
        auto* tc = blob->template GetMutable<Tensor<Context>>();
        if (tc->size() == 0) {
          tc->Resize(paramShape);
          tc->template mutable_data<float>();
        } else {
          CAFFE_ENFORCE(tc->dims() == paramShape);
        }
        lua_pop(L, 1);
        i++;
      }
      CAFFE_ENFORCE_EQ(i, paramBlobs.size());
      lua_pop(L, 2);
    }
    lua_getfield(L, -1, "output");
    if (outputBlobs.size() == 0) {
    } else if (outputBlobs.size() == 1) {
      CAFFE_ENFORCE(
          luaT_isudata(L, -1, torch::Torch<Context>::Traits::tensorTy));
      auto* output =
          static_cast<typename torch::Torch<Context>::Traits::Tensor*>(
              luaT_toudata(L, -1, torch::Torch<Context>::Traits::tensorTy));
      auto outputShape = state_.tensorShape(output);
      auto* blob = outputBlobs[0];
      auto* tc = blob->template GetMutable<Tensor<Context>>();
      tc->Resize(outputShape);
      tc->template mutable_data<float>();
    } else {
      lua_pushnil(L);
      auto i = 0;
      while (lua_next(L, -2) && i < outputBlobs.size()) {
        CAFFE_ENFORCE(
            luaT_isudata(L, -1, torch::Torch<Context>::Traits::tensorTy));
        auto* output =
            static_cast<typename torch::Torch<Context>::Traits::Tensor*>(
                luaT_toudata(L, -1, torch::Torch<Context>::Traits::tensorTy));
        auto outputShape = state_.tensorShape(output);
        auto* blob = outputBlobs[i];
        auto* tc = blob->template GetMutable<Tensor<Context>>();
        if (tc->size() == 0) {
          tc->Resize(outputShape);
          tc->template mutable_data<float>();
        } else {
          CAFFE_ENFORCE(tc->dims() == outputShape);
        }
        ++i;
      }
      CAFFE_ENFORCE_EQ(i, outputBlobs.size());
    }
    lua_pop(L, 2);
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);

    cachedInputSizes_.clear();
    for (const auto* blob : inputBlobs) {
      const auto& dims = blob->template Get<Tensor<Context>>().dims();
      cachedInputSizes_.push_back(dims);
    }
  }

 protected:
  torch::Torch<Context> state_;
  int self_{0};
  std::vector<std::vector<TIndex>> cachedInputSizes_;
};

template <typename Context>
class TorchOp : public TorchOpBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using OperatorBase::Outputs;
  using OperatorBase::Inputs;
  using TorchOpBase<Context>::state_;
  using TorchOpBase<Context>::self_;

  using TorchOpBase<Context>::TorchOpBase;

  bool RunOnDevice() final {
    const auto numInputs =
        OperatorBase::GetSingleArgument<int>("num_inputs", 1);
    const auto numParams =
        OperatorBase::GetSingleArgument<int>("num_params", 0);
    const auto numOutputs =
        OperatorBase::GetSingleArgument<int>("num_outputs", 1);
    CAFFE_ENFORCE_EQ(InputSize(), numInputs + numParams);
    CAFFE_ENFORCE_EQ(OutputSize(), numOutputs);

    std::vector<Blob*> inputBlobs;
    for (auto i = 0; i < numInputs; ++i) {
      inputBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    std::vector<Blob*> paramBlobs;
    for (auto i = numInputs; i < numInputs + numParams; ++i) {
      paramBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    // Outputs must already be pre-sized
    this->reshapeBlobs(inputBlobs, paramBlobs, Outputs());

    lua_State* L = state_.L();
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);
    state_.setContext(&context_);

    // Deserialize self table
    lua_rawgeti(L, LUA_REGISTRYINDEX, self_);

    auto torchOutputs = state_.pushTable(Outputs());
    // set the output field
    lua_setfield(L, -2, "output");
    // set the parameters
    if (numParams != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, 1, 0);
      CAFFE_ENFORCE_EQ(err, 0);
      // iterate the parameters table to put tblobs inside
      lua_pushnil(L);
      auto i = 0;
      while (lua_next(L, -2) && i < numParams) {
        CAFFE_ENFORCE(
            luaT_isudata(L, -1, state_.tensorTy(*paramBlobs[i])),
            luaT_typename(L, -1));
        auto* udata = luaT_toudata(L, -1, state_.tensorTy(*paramBlobs[i]));
        state_.setTensor(
            static_cast<typename torch::Torch<Context>::Traits::Tensor*>(udata),
            const_cast<Blob*>(paramBlobs[i]));
        i++;
        lua_pop(L, 1);
      }
      CAFFE_ENFORCE_EQ(i, numParams);
      lua_pop(L, 1); // pop the parameter table
    }
    // call updateOutput
    // | self
    lua_getfield(L, -1, "updateOutput");
    // | self | updateOutput
    lua_pushvalue(L, -2);
    // | self | updateOutput | self
    auto torchInputs = state_.pushTable(inputBlobs);
    // | self | updateOutput | self | inputs
    int err = lua_pcall(L, 2, 1, 0); // doesn't need the output
    CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L, -1));
    state_.verifyOutputs(Outputs(), torchOutputs);
    lua_pop(L, 2);
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);
    return true;
  }
};

template <typename Context>
class TorchInitOp : public TorchOpBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using OperatorBase::Outputs;
  using OperatorBase::Inputs;
  using TorchOpBase<Context>::TorchOpBase;

  bool RunOnDevice() final {
    const auto numInputs =
        OperatorBase::GetSingleArgument<int>("num_inputs", 1);
    const auto numParams =
        OperatorBase::GetSingleArgument<int>("num_params", 0);
    const auto numOutputs =
        OperatorBase::GetSingleArgument<int>("num_outputs", 1);
    std::vector<Blob*> inputBlobs;
    for (auto i = 0; i < numInputs; ++i) {
      inputBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    std::vector<Blob*> paramBlobs;
    for (auto i = numInputs; i < numInputs + numParams; ++i) {
      paramBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    this->reshapeBlobs(inputBlobs, paramBlobs, Outputs());
    return true;
  }
};

template <typename Context>
class TorchGradientOp : public TorchOpBase<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  using OperatorBase::Outputs;
  using OperatorBase::Inputs;
  using TorchOpBase<Context>::state_;
  using TorchOpBase<Context>::self_;
  using TorchOpBase<Context>::TorchOpBase;

  bool RunOnDevice() final {
    const auto numInputs =
        OperatorBase::GetSingleArgument<int>("num_inputs", 1);
    const auto numParams =
        OperatorBase::GetSingleArgument<int>("num_params", 0);
    const auto numOutputs =
        OperatorBase::GetSingleArgument<int>("num_outputs", 1);
    lua_State* L = state_.L();
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);
    // inputs, params, outputs, grad outputs
    CAFFE_ENFORCE_EQ(InputSize(), numInputs + numParams + 2 * numOutputs);
    // grad inputs, grad params
    CAFFE_ENFORCE_EQ(OutputSize(), numInputs + numParams);
    state_.setContext(&context_);

    std::vector<Blob*> outputBlobs;
    for (auto i = numInputs + numParams; i < numInputs + numParams + numOutputs;
         ++i) {
      outputBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    std::vector<Blob*> inputBlobs;
    for (auto i = 0; i < numInputs; ++i) {
      inputBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    std::vector<Blob*> gradOutputBlobs;
    for (auto i = numInputs + numParams + numOutputs;
         i < numInputs + numParams + numOutputs + numOutputs;
         ++i) {
      gradOutputBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    std::vector<Blob*> gradInputBlobs;
    for (auto i = 0; i < numInputs; ++i) {
      gradInputBlobs.push_back(Outputs()[i]);
    }
    std::vector<Blob*> paramBlobs;
    for (auto i = numInputs; i < numInputs + numParams; ++i) {
      paramBlobs.push_back(const_cast<Blob*>(Inputs()[i]));
    }
    std::vector<Blob*> gradParamBlobs;
    for (auto i = numInputs; i < numInputs + numParams; ++i) {
      gradParamBlobs.push_back(Outputs()[i]);
    }

    // Ensure shapes are correct.
    for (auto i = 0; i < OutputSize(); ++i) {
      Output(i)->ResizeLike(Input(i));
      Output(i)->template mutable_data<float>();
    }

    lua_rawgeti(L, LUA_REGISTRYINDEX, self_);
    state_.pushTable(outputBlobs);
    lua_setfield(L, -2, "output");

    const auto& torchGradInputs = state_.pushTable(gradInputBlobs);
    lua_setfield(L, -2, "gradInput");
    if (numParams != 0) {
      // get the parameters into the stack
      lua_getfield(L, -1, "parameters");
      lua_pushvalue(L, -2);
      int err = lua_pcall(L, 1, LUA_MULTRET, 0);
      CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L, -1));
      // iterate the parameters table to put tblobs inside
      lua_pushnil(L);
      auto i = 0;
      while (lua_next(L, -3) && i < numParams) {
        CAFFE_ENFORCE(luaT_isudata(L, -1, state_.tensorTy(*paramBlobs[i])));
        auto* udata = luaT_toudata(L, -1, state_.tensorTy(*paramBlobs[i]));
        state_.setTensor(
            static_cast<typename torch::Torch<Context>::Traits::Tensor*>(udata),
            const_cast<Blob*>(paramBlobs[i]));
        i++;
        lua_pop(L, 1);
      }
      CAFFE_ENFORCE_EQ(i, numParams);
      // iterate the grad of params
      lua_pushnil(L);
      i = 0;
      while (lua_next(L, -2) && i < numParams) {
        CAFFE_ENFORCE(luaT_isudata(L, -1, state_.tensorTy(*gradParamBlobs[i])));
        auto* udata = luaT_toudata(L, -1, state_.tensorTy(*gradParamBlobs[i]));
        state_.setTensor(
            static_cast<typename torch::Torch<Context>::Traits::Tensor*>(udata),
            const_cast<Blob*>(gradParamBlobs[i]));
        i++;
        lua_pop(L, 1);
      }
      CAFFE_ENFORCE_EQ(i, numParams);
      lua_pop(L, 2); // pop the parameters
    }
    lua_getfield(L, -1, "zeroGradParameters");
    lua_pushvalue(L, -2);
    CAFFE_ENFORCE_EQ(lua_pcall(L, 1, 0, 0), 0);
    state_.pushTable(inputBlobs);
    state_.pushTable(gradOutputBlobs);
    // call
    lua_getfield(L, -3, "accGradParameters");
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    lua_pushnumber(L, 1);
    int err = lua_pcall(L, 4, 0, 0); // doesn't need the output
    CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L, -1));
    lua_getfield(L, -3, "updateGradInput");
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    lua_pushvalue(L, -4);
    err = lua_pcall(L, 3, 1, 0); // doesn't need the output
    CAFFE_ENFORCE_EQ(err, 0, lua_tostring(L, -1));
    state_.verifyOutputs(gradInputBlobs, torchGradInputs);
    lua_pop(L, 4);
    CAFFE_ENFORCE_EQ(lua_gettop(L), 0);
    return true;
  }
};
}
