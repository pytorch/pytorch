#include "torch/csrc/autograd/c2isl_function.h"
#include "torch/csrc/autograd/functions/utils.h"

#include "torch/csrc/utils/auto_gil.h"
#include "torch/csrc/utils/auto_gpu.h"
#include "torch/csrc/utils/object_ptr.h"

// TODO: remove me when we have actual backwards
#include "torch/csrc/autograd/functions/basic_ops.h"

#include "THC/THC.h"

extern THCState* state;

namespace torch { namespace autograd {

DLDataType toDLDataType(at::Type& ty) { // should be const
  DLDataType t;
  // Torch does not have any vectorized types.
  t.lanes = 1;
  switch (ty.scalarType()) {
    case at::ScalarType::Byte:
      t.code = DLDataTypeCode::kUInt;
      t.bits = 8;
      break;
    case at::ScalarType::Char:
      t.code = DLDataTypeCode::kInt;
      t.bits = 8;
      break;
    case at::ScalarType::Double:
      t.code = DLDataTypeCode::kFloat;
      t.bits = 64;
      break;
    case at::ScalarType::Float:
      t.code = DLDataTypeCode::kFloat;
      t.bits = 32;
      break;
    case at::ScalarType::Int:
      t.code = DLDataTypeCode::kInt;
      t.bits = 32;
      break;
    case at::ScalarType::Long:
      t.code = DLDataTypeCode::kInt;
      t.bits = 64;
      break;
    case at::ScalarType::Short:
      t.code = DLDataTypeCode::kInt;
      t.bits = 16;
      break;
    case at::ScalarType::Half:
      t.code = DLDataTypeCode::kFloat;
      t.bits = 16;
      break;
    case at::ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
  return t;
}

tvm::Type toTVMType(at::ScalarType s) {
  switch (s) {
    case at::ScalarType::Byte:    return tvm::UInt(8);
    case at::ScalarType::Char:    return tvm::Int(8);
    case at::ScalarType::Double:  return tvm::Float(64);
    case at::ScalarType::Float:   return tvm::Float(32);
    case at::ScalarType::Int:     return tvm::Int(32);
    case at::ScalarType::Long:    return tvm::Int(64);
    case at::ScalarType::Short:   return tvm::Int(16);
    case at::ScalarType::Half:    return tvm::Float(16);
    case at::ScalarType::NumOptions:
      throw std::logic_error("NumOptions is not a valid ScalarType");
  }
}

tvm::Type toTVMType(const DLDataType& t) {
  switch (t.code) {
    case DLDataTypeCode::kUInt:  return tvm::UInt (t.bits, t.lanes);
    case DLDataTypeCode::kInt:   return tvm::Int  (t.bits, t.lanes);
    case DLDataTypeCode::kFloat: return tvm::Float(t.bits, t.lanes);
    default:
      throw std::logic_error("Unknown type code " + std::to_string(t.code));
  }
}

// Types never die
at::Type& toATenType(at::Backend backend, const DLDataType& t) {
  at::ScalarType st;
  switch (t.code) {
    case DLDataTypeCode::kUInt:
      switch (t.bits) {
        case 8:  st = at::ScalarType::Byte;   break;
        default: throw std::logic_error("Unsupported kUInt bits " + std::to_string(t.bits));
      }
      break;
    case DLDataTypeCode::kInt:
      switch (t.bits) {
        case 8:  st = at::ScalarType::Char;   break;
        case 16: st = at::ScalarType::Short;  break;
        case 32: st = at::ScalarType::Int;    break;
        case 64: st = at::ScalarType::Long;   break;
        default: throw std::logic_error("Unsupported kInt bits " + std::to_string(t.bits));
      }
      break;
    case DLDataTypeCode::kFloat:
      switch (t.bits) {
        case 16: st = at::ScalarType::Half;   break;
        case 32: st = at::ScalarType::Float;  break;
        case 64: st = at::ScalarType::Double; break;
        default: throw std::logic_error("Unsupported kFloat bits " + std::to_string(t.bits));
      }
      break;
    default:
      throw std::logic_error("Unsupported code " + std::to_string(t.code));
  }
  // TODO: This doesn't put the tensor on the correct device
  return getType(backend, st);
}

at::Type& toATenType(const DLContext& ctx, const DLDataType& t) {
  if (t.lanes != 1) throw std::logic_error("ATen does not support lanes != 1");
  at::Backend backend;
  switch (ctx.device_type) {
    case DLDeviceType::kCPU:
      backend = at::Backend::CPU;
    case DLDeviceType::kGPU:
      backend = at::Backend::CUDA;
    default:
      throw std::logic_error("Unsupported device_type " + std::to_string(ctx.device_type));
  }
  return toATenType(backend, t);
}

// NB: You're not allowed to read off the backend from DLMetadata,
// it won't be initialized!
at::Tensor newATenTensor(at::Backend backend, const DLMetadata& meta) {
  return toATenType(backend, meta.dtype)
            .tensor(at::IntList(meta.shape,   meta.ndim),
                    at::IntList(meta.strides, meta.ndim));
}

// Allocates!  Because ATen doesn't let us take a non-const
// pointer to underlying data, but dlpack doesn't promise
// anything about const-ness.
int64_t* newDLInt64Array(const at::IntList& arr) {
  auto len = arr.size();
  auto r = new int64_t[len];
  for (size_t i = 0; i < len; i++) {
    r[i] = arr[i];
  }
  return r;
}

DLMetadataUPtr toDLMetadata(at::Tensor& t) { // should be const
  DLMetadataUPtr res(new DLMetadata);
  res->ndim = t.dim();
  res->dtype = toDLDataType(t.type());
  res->shape = newDLInt64Array(t.sizes()); // ownership transfer
  res->strides = newDLInt64Array(t.strides()); // ownership transfer
  // invalid values which should not be used
  res->data = nullptr;
  res->ctx.device_type = DLDeviceType(0);
  res->ctx.device_id = -1;
  res->byte_offset = 0;
  return res;
}

// NB: DLTensorUPtr is only live as long as Tensor is
// TODO: check ownership against numpy (in cwrap)
DLTensorUPtr toDLTensor(at::Tensor& t) {
  DLTensorUPtr res(new DLTensor, DLTensorDeleter(t));
  // NB: It is NOT sound to resize afterwards.  It would be good to clear
  // TH_STORAGE_RESIZABLE but ATen does not currently expose this
  // functionality, see https://github.com/zdevito/ATen/issues/28
  res->data = t.data_ptr();
  if (t.type().isCuda()) {
    res->ctx.device_type = DLDeviceType::kGPU;
    res->ctx.device_id = t.get_device();
  } else {
    res->ctx.device_type = DLDeviceType::kCPU;
    // TODO: It's not clear what the semantics of device_id are supposed
    // to be on CPU
    res->ctx.device_id = 0;
  }
  res->ndim = t.dim();
  res->dtype = toDLDataType(t.type());
  // It would be nice to directly use the data() of these
  // objects, but ATen is const and dlpack is non-const, which
  // means a client could scribble over the data. Safer to not.
  res->shape = newDLInt64Array(t.sizes()); // ownership transfer
  res->strides = newDLInt64Array(t.strides()); // ownership transfer
  // NB: byte_offset is already applied by THTensor_(data), which is what backs
  // data_ptr.
  res->byte_offset = 0;
  return res;
}

// TODO: Replace this with the direct thing when ATen replaces thpp
at::Tensor variableToATen(const std::shared_ptr<Variable>& var) {
  THPObjectPtr pyvar(THPVariable_Wrap(var));
  PyObject* data = ((THPVariable*)pyvar.get())->data;
  return createTensorAT(data);
}

variable_list IslFunction::apply(const variable_list& input_vars) {
  AutoGIL gil;

  // Use the same backend as the first tensor
  auto& input = input_vars.at(0);
  AutoGPU guard(input->data->getDevice());

  if (!pImpl_) {
    std::vector<const DLMetadata*> inMetas; // ownership managed by inTensorUPtrs
    std::vector<DLMetadataUPtr> inMetaUPtrs;
    auto inputs_it = inputs.begin();
    for (auto input : input_vars) {
      auto at = variableToATen(input);
      inMetaUPtrs.emplace_back(toDLMetadata(at));
      auto meta = inMetaUPtrs.back().get();
      inMetas.emplace_back(meta);
      // Check for consistency with TVM description
      if ((*inputs_it)->dtype.code() != meta->dtype.code ||
          (*inputs_it)->dtype.bits() != meta->dtype.bits ||
          (*inputs_it)->dtype.lanes() != meta->dtype.lanes) {
        throw std::logic_error("input tensor incompatible with tvm types");
      }
      inputs_it++;
    }
    pImpl_ = std::unique_ptr<c2isl::ISLTVMIROp>(
              new c2isl::ISLTVMIROp(outputs, inputs, vars, ops)
             );
    pImpl_->SetKernelName(kernelName);
    pImpl_->SetKernelOptions(islKernelOptions);
    outputDLMetas_ = pImpl_->JITCompile(inMetas);
  }

  // TODO: check size consistency

  auto backend = variableToATen(input).type().backend();
  std::vector<at::Tensor> output_ats;
  for (auto& i : outputDLMetas_) {
    output_ats.emplace_back(newATenTensor(backend, *i));
  }

  std::vector<int> P;
  for (auto& e : pImpl_->tvmActualParams_) {
    P.push_back(e.as<tvm::ir::IntImm>()->value);
  }

  std::vector<void*> O;
  for (auto& output : output_ats) {
    O.push_back(output.data_ptr());
  }

  std::vector<const void*> I;
  for (auto& input : input_vars) {
    I.push_back(variableToATen(input).data_ptr());
  }

  auto& grids = pImpl_->GetGridDims();
  auto& blocks = pImpl_->GetBlockDims();

  pImpl_->rtcFun_.Launch(
    grids[0], grids[1], grids[2],
    blocks[0], blocks[1], blocks[2],
    0, THCState_getCurrentStream(state),
    P, O, I);

  // TODO: Remove this idiotic conversion once ATen replace thpp
  // in PyTorch proper.  Once you do, remove AutoGIL, since this
  // is the only place we're interacting with Python.
  tensor_list compat_outputs;
  for (auto& output : output_ats) {
    THPObjectPtr p(createPyObject(output));
    compat_outputs.emplace_back(createTensor(p.get()));
  }

  // OK, return outputs
  return wrap_outputs(input_vars, std::move(compat_outputs), [&](FunctionFlags f) -> std::shared_ptr<Function> {
    return std::make_shared<Error>("IslFunction backwards not implemented yet", std::move(f));
  });
}

    /*
void IslMatMul::initTVM(const std::vector<const DLMetadata*>& input_metas) {
  // TODO: I feel dirty doing copy-assignment
  vars = { ::tvm::Var("M"), ::tvm::Var("N"), ::tvm::Var("O") };
  inputs = {
    ::tvm::placeholder({vars[0],vars[1]}, toTVMType(input_metas[0]->dtype), "A"),
    ::tvm::placeholder({vars[1],vars[2]}, toTVMType(input_metas[1]->dtype), "B")
  };
  ops = {
    ::tvm::makeMatMult(inputs[0], inputs[1], "C", 0, 0
      // TODO: make the transpose configurable
      //OperatorBase::GetSingleArgument<int>("trans_a", 0),
      //OperatorBase::GetSingleArgument<int>("trans_b", 0)
    )
  };
  outputs = { ops[0] };
  kernelName = "matmul";
  islKernelOptions.strategy = c2isl::ISLStrategy::Strategy::Default;
  islKernelOptions.tilingAnnotation = ""; // use default PPCG
}
    // TODO GRADIENT!
    this->SetupGradient([](const ArgumentHelper& args, std::vector<::tvm::Tensor> I, std::vector<::tvm::Tensor> GO, std::vector<std::string> GI) {
      bool trans_a = args.GetSingleArgument<int>("trans_a", 0);
      bool trans_b = args.GetSingleArgument<int>("trans_b", 0);
      std::vector<::tvm::Tensor> gops;

      if (trans_a) {
        if (trans_b) {
          gops.emplace_back(::tvm::makeMatMult(I[1], GO[0], GI[0], true, true));
          gops.emplace_back(::tvm::makeMatMult(GO[0], I[0], GI[1], true, true));
        } else {
          gops.emplace_back(::tvm::makeMatMult(I[1], GO[0], GI[0], false, true));
          gops.emplace_back(::tvm::makeMatMult(I[0], GO[0], GI[1], false, false));
        }
      } else {
        if (trans_b) {
          gops.emplace_back(::tvm::makeMatMult(GO[0], I[1], GI[0], false, false));
          gops.emplace_back(::tvm::makeMatMult(GO[0], I[0], GI[1], true, false));
        } else {
          gops.emplace_back(::tvm::makeMatMult(GO[0], I[1], GI[0], false, true));
          gops.emplace_back(::tvm::makeMatMult(I[0], GO[0], GI[1], true, false));
        }
      }
      std::vector<::tvm::Tensor> goutput{ gops[0], gops[1] };
      return std::make_pair(gops, goutput);
    }
    */

}} // namespace torch::autograd
