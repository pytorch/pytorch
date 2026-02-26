#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/distributed/autograd/autograd.h>
#include <torch/csrc/distributed/autograd/rpc_messages/rref_backward_req.h>
#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch::distributed::rpc {

/////////////////////  Pickle/Unpickle Helplers ////////////////////////////

namespace {

py::tuple toPyTuple(const RRefForkData& rrefForkData) {
  // add GIL as it is constructing a py::object
  pybind11::gil_scoped_acquire ag;
  return py::make_tuple(
      rrefForkData.ownerId_,
      rrefForkData.rrefId_.createdOn_,
      rrefForkData.rrefId_.localId_,
      rrefForkData.forkId_.createdOn_,
      rrefForkData.forkId_.localId_,
      rrefForkData.parent_,
      rrefForkData.typeStr_);
}

RRefForkData fromPyTuple(const py::tuple& pyTuple) {
  // add GIL as it is accessing a py::object
  pybind11::gil_scoped_acquire ag;
  TORCH_INTERNAL_ASSERT(
      pyTuple.size() == RFD_TUPLE_SIZE,
      "Pickled RRefForkData must contain ",
      RFD_TUPLE_SIZE,
      " numbers.");
  worker_id_t ownerId = pyTuple[OWNER_IDX].cast<worker_id_t>();
  // const reference will extend the lifetime of the temporary variable
  const RRefId& rrefId = RRefId(
      pyTuple[RREFID_ON_IDX].cast<worker_id_t>(),
      pyTuple[RREFID_ID_IDX].cast<local_id_t>());
  const RRefId& forkId = RRefId(
      pyTuple[FORKID_ON_IDX].cast<worker_id_t>(),
      pyTuple[FORKID_ID_IDX].cast<local_id_t>());

  worker_id_t parent = pyTuple[PARENT_IDX].cast<worker_id_t>();
  const std::string& typeStr = pyTuple[TYPE_IDX].cast<std::string>();

  return RRefForkData(ownerId, rrefId, forkId, parent, typeStr);
}

TypePtr tryInferTypeWithTypeHint(
    const py::object& value,
    const py::object& type_hint) {
  // If the py::object to be contained by the RRef is a ScriptModule, we enforce
  // users to specify its ModuleInterface type.
  if (auto module = jit::as_module(value)) {
    TORCH_CHECK(
        !type_hint.is_none(),
        "The RRef being created contains a ScriptModule, "
        "must provide its ModuleInterface type hint. ");
    c10::QualifiedName type_qualified_name = c10::QualifiedName(
        py::cast<std::string>(py::module::import("torch._jit_internal")
                                  .attr("_qualified_name")(type_hint)));
    TypePtr type_hint_ptr =
        jit::get_python_cu()->get_interface(type_qualified_name);
    std::ostringstream subtype_check_msg;
    TORCH_CHECK(
        type_hint_ptr != nullptr &&
            module.value().type()->isSubtypeOfExt(
                *type_hint_ptr, &subtype_check_msg),
        module.value().type()->repr_str(),
        " is not a subtype of the type hint: ",
        type_qualified_name.qualifiedName(),
        ", did you pass a valid interface type?\n",
        subtype_check_msg.str());
    return type_hint_ptr;
  } else {
    TORCH_CHECK(
        type_hint.is_none(),
        "type_hint should only be specified when the RRef being created contains a ScriptModule.");
  }

  // Check if value is an instance of a ScriptClass. If not, skip type inference
  // because it will try to script the class that value is in instance of, and
  // this should be avoided.
  py::bool_ can_compile =
      py::module::import("torch._jit_internal")
          .attr("can_compile_class")(py::type::handle_of(value));

  if (py::cast<bool>(can_compile)) {
    py::object existing_ty =
        py::module::import("torch.jit._state")
            .attr("_get_script_class")(py::type::handle_of(value));

    if (existing_ty.is_none()) {
      return PyObjectType::get();
    }
  }

  // NB: `jit::tryToInferType(..)` infers types including ScriptClass, but
  // excluding ScriptModule.
  jit::InferredType type_inferred = jit::tryToInferType(value);
  if (type_inferred.success()) {
    // If we could infer the type from the pyobject, we create
    // the RRef with the IValue of that type.
    return type_inferred.type();
  }

  // Otherwise it's a pure pyobject, create the RRef
  // that holds an IValue of an pyobject.
  return PyObjectType::get();
}

} // namespace

///////////////////////////  PyRRef  //////////////////////////////////

PyRRef::PyRRef(c10::intrusive_ptr<RRef> rref)
    : rref_(std::move(rref)), profilingFuture_(std::nullopt) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
  C10_LOG_API_USAGE_ONCE("torch.distributed.rref");
}

PyRRef::PyRRef(const py::object& value, const py::object& type_hint)
    : PyRRef([&value, &type_hint]() mutable {
        TypePtr elem_type = tryInferTypeWithTypeHint(value, type_hint);
        auto rref = RRefContext::getInstance().createOwnerRRef(elem_type);
        // jit::toIValue takes a py::handle as the first argument, and it calls
        // py::handle.cast<py::object>() to incref of provided value. The
        // returned ivalue will keep the reference alive.
        // NB: the first argument const py::object& value must be kept alive
        // until the following jit::toIValue returns (i.e., incref done). That's
        // why this ctor can only be called while holding GIL.
        IValue ivalue = jit::toIValue(value, elem_type);
        rref->setValue(std::move(ivalue));
        return rref;
      }()) {}

// NOLINTNEXTLINE(bugprone-exception-escape)
PyRRef::~PyRRef() {
  if (type_.has_value()) {
    pybind11::gil_scoped_acquire ag;
    (*type_).dec_ref();
    // explicitly setting PyObject* to nullptr to prevent py::object's dtor to
    // decref on the PyObject again.
    // See Note [Destructing py::object] in python_ivalue.h
    (*type_).ptr() = nullptr;
  }
}

c10::intrusive_ptr<JitFuture> PyRRef::getFuture() const {
  // Marking hasValue to false, as this Future is only used for signaling
  // profiler to update profiling result and the profiler does not retrieve
  // any value from it.
  return toPyJitFuture(rref_->getOwnerCreationFuture(), false /* hasValue */);
}

c10::intrusive_ptr<JitFuture> PyRRef::getProfilingFuture() const {
  TORCH_INTERNAL_ASSERT(profilingFuture_, "Profiling future has not been set!");
  return *profilingFuture_;
}

void PyRRef::setProfilingFuture(c10::intrusive_ptr<JitFuture> profilingFuture) {
  profilingFuture_ = std::move(profilingFuture);
}

bool PyRRef::isOwner() const {
  return rref_->isOwner();
}

bool PyRRef::confirmedByOwner() const {
  return rref_->confirmedByOwner();
}

WorkerInfo PyRRef::owner() const {
  return RRefContext::getInstance().agent()->getWorkerInfo(rref_->owner());
}

std::string PyRRef::ownerName() const {
  return rref_->ownerName();
}

py::object PyRRef::toHere(const float timeoutSeconds) const {
  C10_LOG_API_USAGE_ONCE("torch.distributed.rref.to_here");
  if (rref_->isOwner()) {
    return localValue();
  } else {
    // toHere() calls python_rpc_handler which acquires GIL when UserRRef holds
    // a python object
    IValue value = c10::static_intrusive_pointer_cast<UserRRef>(rref_)->toHere(
        timeoutSeconds);

    if (rref_->isPyObj()) {
      // python_rpc_handler deserialization will acquires GIL.
      auto rfr_values = value.toTupleRef().elements().vec();
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      auto ret = pythonRpcHandler.deserialize(
          SerializedPyObj::fromIValues(std::move(rfr_values)));
      pythonRpcHandler.handleException(ret);
      return ret;
    } else {
      // acquiring GIL as torch::jit::toPyObject creates new py::object
      // without grabbing the GIL.
      pybind11::gil_scoped_acquire ag;
      return torch::jit::toPyObject(std::move(value));
    }
  }
}

py::object PyRRef::localValue() const {
  TORCH_CHECK(
      rref_->isOwner(),
      "For ",
      *rref_,
      ", can't call localValue() on user ",
      RRefContext::getInstance().agent()->getWorkerInfo(),
      ". Call it on owner ",
      owner());

  py::object res;
  auto value =
      c10::static_intrusive_pointer_cast<const OwnerRRef>(rref_)->getValue();
  auto& rpcHandler = PythonRpcHandler::getInstance();
  {
    // acquiring GIL as torch::jit::toPyObject creates new py::object without
    // grabbing the GIL.
    pybind11::gil_scoped_acquire ag;
    res = torch::jit::toPyObject(std::move(value));
    rpcHandler.handleExceptionGILHeld(res);
  }
  return res;
}

std::string PyRRef::str() const {
  if (rref_->isOwner()) {
    return c10::str("OwnerRRef(", rref_->rrefId(), ")");
  } else {
    return c10::str(
        "UserRRef(RRefId = ",
        rref_->rrefId(),
        ", ForkId = ",
        c10::static_intrusive_pointer_cast<UserRRef>(rref_)->forkId(),
        ")");
  }
}

py::object PyRRef::createRRefProxy(
    const RRefProxyType& type,
    float timeoutSeconds) const {
  auto& pythonRpcHandler = PythonRpcHandler::getInstance();
  pybind11::gil_scoped_acquire ag;
  auto& functions = pythonRpcHandler.getRRefProxyFunctions();
  auto& ctor = functions.rrefProxyCtor_;
  switch (type) {
    case RRefProxyType::RPC_SYNC: {
      return ctor(*this, functions.rpcSync_, timeoutSeconds);
    }
    case RRefProxyType::RPC_ASYNC: {
      return ctor(*this, functions.rpcAsync_, timeoutSeconds);
    }
    case RRefProxyType::REMOTE: {
      return ctor(*this, functions.remote_, timeoutSeconds);
    }
    default: {
      TORCH_INTERNAL_ASSERT(false, "Unrecognized RRefProxy type ", type);
    }
  }
}

py::object PyRRef::getRRefType(float timeout, bool blocking) {
  // GIL is not released when calling this function.
  if (!type_.has_value()) {
    pybind11::gil_scoped_release release;
    auto& pythonRpcHandler = PythonRpcHandler::getInstance();
    auto& typeFuncs = pythonRpcHandler.getRRefTypeFunctions();
    pybind11::gil_scoped_acquire acquire;
    type_ = isOwner() ? typeFuncs.onOwner_(*this, blocking)
                      : typeFuncs.onUser_(*this, timeout, blocking);
  }
  // Returns py::object that can be Python type or future.
  return *type_;
}

py::tuple PyRRef::pickle() const {
  auto& ctx = RRefContext::getInstance();
  auto rrefForkData = ctx.prepareChildFork(rref_);
  return toPyTuple(rrefForkData);
}

PyRRef PyRRef::unpickle(const py::tuple& pyTuple) {
  auto& ctx = RRefContext::getInstance();
  auto rrefForkData = fromPyTuple(pyTuple);
  TypePtr rrefType =
      PythonRpcHandler::getInstance().parseTypeFromStr(rrefForkData.typeStr_);
  c10::intrusive_ptr<RRef> rref = ctx.getOrCreateRRef(rrefForkData, rrefType);
  ctx.notifyOwnerAndParentOfFork(
      rrefForkData.forkId_, rrefForkData.parent_, rref);
  return PyRRef(std::move(rref));
}

c10::IValue PyRRef::toIValue() const {
  // cast to RRefInterface to hold it into IValue
  auto rrefPtr = c10::static_intrusive_pointer_cast<c10::RRefInterface>(rref_);
  return IValue(rrefPtr);
}

void PyRRef::backward(int64_t autogradContextId, bool retainGraph) {
  backward(autogradContextId, retainGraph, rref_);
}

void PyRRef::backwardOwnerRRef(
    int64_t autogradContextId,
    bool retainGraph,
    IValue value) {
  // If we have a PyObj, retrieve the underlying tensor.
  if (value.isPyObject()) {
    py::gil_scoped_acquire gil;
    py::object obj = torch::jit::toPyObject(value);
    try {
      value = torch::jit::toIValue(obj, c10::TensorType::get());
    } catch (py::cast_error&) {
      TORCH_CHECK(false, "RRef should contain a tensor for .backward()");
    }
  }

  TORCH_CHECK(value.isTensor(), "RRef should contain a tensor for .backward()");
  auto root = value.toTensor();

  if (autogradContextId == -1) {
    torch::autograd::backward({root});
  } else {
    torch::distributed::autograd::backward(
        autogradContextId, {root}, retainGraph);
  }
}

void PyRRef::backward(
    int64_t autogradContextId,
    bool retainGraph,
    const c10::intrusive_ptr<RRef>& rref) {
  if (rref->isOwner()) {
    backwardOwnerRRef(
        autogradContextId,
        retainGraph,
        c10::static_intrusive_pointer_cast<const OwnerRRef>(rref)->getValue());
  } else {
    TORCH_CHECK(
        autogradContextId != -1,
        "User RRefs require 'dist_autograd_ctx_id' to be specified");

    autograd::RRefBackwardReq rrefBackwardReq(
        rref->rrefId(), autogradContextId, retainGraph);

    // Invoke distributed backward remotely.
    auto rpcAgent = rpc::RpcAgent::getCurrentRpcAgent();
    rpcAgent
        ->send(
            rpcAgent->getWorkerInfo(rref->owner()),
            std::move(rrefBackwardReq).toMessage())
        ->waitAndThrow();
  }
}

} // namespace torch::distributed::rpc
