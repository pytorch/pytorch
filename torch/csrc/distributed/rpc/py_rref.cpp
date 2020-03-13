#include <torch/csrc/distributed/rpc/py_rref.h>

#include <torch/csrc/distributed/rpc/python_functions.h>
#include <torch/csrc/distributed/rpc/python_rpc_handler.h>
#include <torch/csrc/distributed/rpc/rref_context.h>
#include <torch/csrc/jit/python/module_python.h>
#include <torch/csrc/jit/python/pybind_utils.h>

namespace torch {
namespace distributed {
namespace rpc {

/////////////////////  Pickle/Unpickle Helplers ////////////////////////////

namespace {
py::tuple toPyTuple(const RRefForkData& rrefForkData) {
  // add GIL as it is contructing a py::object
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
  // If the py::object to be contained by the RRef is a ScripModule, we enforce
  // users to specify its ModuleInterface type.
  if (auto module = jit::as_module(value)) {
    TORCH_CHECK(
        !type_hint.is_none(),
        "The RRef being created contains a ScriptModule, "
        "must provide its ModuleInterface type hint. ");
    c10::QualifiedName type_qualified_name = c10::QualifiedName(
        py::cast<std::string>(py::module::import("torch.jit")
                                  .attr("_qualified_name")(type_hint)));
    TypePtr type_hint_ptr =
        jit::get_python_cu()->get_interface(type_qualified_name);
    TORCH_CHECK(
        type_hint_ptr != nullptr &&
            module.value().type()->isSubtypeOf(type_hint_ptr),
        module.value().type()->python_str(),
        " is not a subtype of the type hint: ",
        type_qualified_name.qualifiedName(),
        ", did you pass a valid interface type?");
    return type_hint_ptr;
  } else {
    TORCH_CHECK(
        type_hint.is_none(),
        "type_hint should only be specified when the RRef being created contains a ScriptModule.");
  }

  // NB: `jit::tryToInferType(..)` infers types including ScriptClass, but
  // excluding ScripModule.
  jit::InferredType type_inferred = jit::tryToInferType(value);
  if (type_inferred.success()) {
    // If we could infer the type from the pyobject, we create
    // the RRef with the IValue of that type.
    return type_inferred.type();
  }

  // Otherwise it's a pure pyobject, create the RRef
  // that holds an IValue of an pyobject.
  return PyObjectType::get();
} // namespace

} // namespace

///////////////////////////  PyRRef  //////////////////////////////////

PyRRef::PyRRef(c10::intrusive_ptr<RRef> rref) : rref_(std::move(rref)) {
  TORCH_CHECK(rref_, "PyRRef must not wrap nullptr");
}

PyRRef::PyRRef(const py::object& value, const py::object& type_hint)
    : PyRRef([&value, &type_hint]() {
        TypePtr elem_type = tryInferTypeWithTypeHint(value, type_hint);
        auto rref = RRefContext::getInstance().createOwnerRRef(elem_type);
        py::object copy(value); // increases refcount
        IValue ivalue = jit::toIValue(std::move(copy), elem_type);
        rref->setValue(std::move(ivalue));
        return rref;
      }()) {}

bool PyRRef::isOwner() const {
  return rref_->isOwner();
}

WorkerInfo PyRRef::owner() const {
  return RRefContext::getInstance().agent()->getWorkerInfo(rref_->owner());
}

py::object PyRRef::toHere() {
  if (rref_->isOwner()) {
    return localValue();
  } else {
    // toHere() calls python_rpc_handler which acquires GIL when UserRRef holds
    // a python object
    IValue value =
        c10::static_intrusive_pointer_cast<UserRRef>(rref_)->toHere();
    if (rref_->isPyObj()) {
      // python_rpc_handler deserialization will acquires GIL.
      auto rfr_values = value.toTuple()->elements();
      auto& pythonRpcHandler = PythonRpcHandler::getInstance();
      auto ret = pythonRpcHandler.deserialize(
          SerializedPyObj::fromIValues(rfr_values));
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

py::object PyRRef::localValue() {
  TORCH_CHECK(
      rref_->isOwner(),
      "Cannot call localValue() on a non-local reference. Call it on ",
      owner().name_);

  py::object res;
  auto value = c10::static_intrusive_pointer_cast<OwnerRRef>(rref_)->getValue();
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
  std::ostringstream ss;
  if (rref_->isOwner()) {
    ss << "OwnerRRef(" << rref_->rrefId() << ")";
  } else {
    ss << "UserRRef(RRefId = " << rref_->rrefId() << ", ForkId = "
       << c10::static_intrusive_pointer_cast<UserRRef>(rref_)->forkId() << ")";
  }
  return ss.str();
}

py::tuple PyRRef::pickle() const {
  auto& ctx = RRefContext::getInstance();
  // TODO: use a dispatch table to pickle/unpickle an RRef, and only only
  // install the dispatch table only when there are indeed RPC activities. As
  // a counter example, checkpointing a model with RRefs should not trigger
  // forks to be added as a fork or a child.
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

c10::IValue PyRRef::toIValue() {
  // cast to RRefInterface to hold it into IValue
  auto rrefPtr = c10::static_intrusive_pointer_cast<c10::RRefInterface>(rref_);
  return IValue(rrefPtr);
}

} // namespace rpc
} // namespace distributed
} // namespace torch
