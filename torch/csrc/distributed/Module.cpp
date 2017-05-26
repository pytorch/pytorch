#include <Python.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "torch/csrc/utils/python_strings.h"
#include "THDP.h"

static std::unordered_map<std::string, THDChannelType> name2channel_type = {
    {"mpi", THDChannelMPI},
    {"tcp", THDChannelTCP},
    {"gloo", THDChannelGloo},
};

static bool THDPModule_loadClasses(PyObject *self)
{
#ifdef WITH_DISTRIBUTED_MW
#define ASSERT_NOT_NULL(ptr) if (!(ptr)) { THPUtils_setError("couldn't load classes"); return false; }
  PyObject *torch_module = PyImport_ImportModule("torch.distributed");
  if (!torch_module) {
    THPUtils_setError("class loader couldn't access torch.distributed module");
    return false;
  }

  if (!THDPDoubleTensor_postInit(torch_module)) return false;
  if (!THDPFloatTensor_postInit(torch_module)) return false;
  //if (!THDPHalfTensor_postInit(torch_module)) return false;
  if (!THDPLongTensor_postInit(torch_module)) return false;
  if (!THDPIntTensor_postInit(torch_module)) return false;
  if (!THDPShortTensor_postInit(torch_module)) return false;
  if (!THDPCharTensor_postInit(torch_module)) return false;
  if (!THDPByteTensor_postInit(torch_module)) return false;

  ASSERT_NOT_NULL(THDPDoubleStorageClass = PyObject_GetAttrString(torch_module,(char*)"DoubleStorage"));
  ASSERT_NOT_NULL(THDPFloatStorageClass  = PyObject_GetAttrString(torch_module,(char*)"FloatStorage"));
  //ASSERT_NOT_NULL(THDPHalfStorageClass   = PyObject_GetAttrString(torch_module,(char*)"HalfStorage"));
  ASSERT_NOT_NULL(THDPLongStorageClass   = PyObject_GetAttrString(torch_module,(char*)"LongStorage"));
  ASSERT_NOT_NULL(THDPIntStorageClass    = PyObject_GetAttrString(torch_module,(char*)"IntStorage"));
  ASSERT_NOT_NULL(THDPShortStorageClass  = PyObject_GetAttrString(torch_module,(char*)"ShortStorage"));
  ASSERT_NOT_NULL(THDPCharStorageClass   = PyObject_GetAttrString(torch_module,(char*)"CharStorage"));
  ASSERT_NOT_NULL(THDPByteStorageClass   = PyObject_GetAttrString(torch_module,(char*)"ByteStorage"));

  return true;
#undef ASSERT_NOT_NULL
#endif
}

static std::unordered_map<PyObject*, THDReduceOp> obj2reduceop;
static std::unordered_map<PyObject*, THDGroup> obj2group;

PyObject* THDPModule_initProcessGroup(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 4 || !THPUtils_checkString(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkString(PyTuple_GET_ITEM(args, 1)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) ||
        !THPUtils_checkString(PyTuple_GET_ITEM(args, 3))) {
    THPUtils_invalidArguments(args, NULL, "init_process_group", 1, "(string backend, string init_method, int world_size, string world_size)");
    return NULL;
  }

  std::string backend_name = THPUtils_unpackString(PyTuple_GET_ITEM(args, 0));
  std::string init_method = THPUtils_unpackString(PyTuple_GET_ITEM(args, 1));
  int world_size = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
  std::string group_name = THPUtils_unpackString(PyTuple_GET_ITEM(args, 1));
  THDChannelType channel_type = name2channel_type.at(backend_name);
  THPUtils_assert(
      THDProcessGroupInit(channel_type, init_method, world_size, group_name),
      "failed to initialize distributed library (THD)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_initMasterWorker(PyObject *_unused, PyObject *backend)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkString(backend),
      "backend argument has to be a string/bytes object, but got %s",
      THPUtils_typename(backend));
  std::string backend_name = THPUtils_unpackString(backend);
  THDChannelType channel_type = name2channel_type.at(backend_name);
  THPUtils_assert(THDMasterWorkerInit(channel_type), "failed to initialize "
      "distributed library (THD)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_getRank(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return PyInt_FromLong(THDGetRank());
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_getNumProcesses(PyObject *_unused)
{
  HANDLE_TH_ERRORS
  return PyInt_FromLong(THDGetNumProcesses());
  END_HANDLE_TH_ERRORS
}

#ifdef WITH_CUDA
extern PyObject* THCPDoubleTensorClass;
extern PyObject* THCPFloatTensorClass;
extern PyObject* THCPHalfTensorClass;
extern PyObject* THCPLongTensorClass;
extern PyObject* THCPIntTensorClass;
extern PyObject* THCPShortTensorClass;
extern PyObject* THCPCharTensorClass;
extern PyObject* THCPByteTensorClass;
#endif

static THDTensorDescriptor* _makeDescriptor(PyObject *obj)
{
  PyObject *type = (PyObject*)Py_TYPE(obj);
#define REGISTER_TH_DESCRIPTOR(TYPE)                                           \
  if (type == THP##TYPE##Class)                                                \
    return THDTensorDescriptor_newFromTH##TYPE(((THP##TYPE*)obj)->cdata);
  REGISTER_TH_DESCRIPTOR(DoubleTensor);
  REGISTER_TH_DESCRIPTOR(FloatTensor);
  REGISTER_TH_DESCRIPTOR(LongTensor);
  REGISTER_TH_DESCRIPTOR(IntTensor);
  REGISTER_TH_DESCRIPTOR(ShortTensor);
  REGISTER_TH_DESCRIPTOR(CharTensor);
  REGISTER_TH_DESCRIPTOR(ByteTensor);
#undef REGISTER_TH_DESCRIPTOR
#ifdef WITH_CUDA
#define REGISTER_THC_DESCRIPTOR(TYPE)                                           \
  if (type == THCP##TYPE##Class)                                                \
    return THDTensorDescriptor_newFromTHCuda##TYPE((THCuda##TYPE*)(((torch::THPVoidTensor*)obj)->cdata));
  REGISTER_THC_DESCRIPTOR(DoubleTensor);
  if (type == THCPFloatTensorClass)
    return THDTensorDescriptor_newFromTHCudaFloatTensor((THCudaTensor*)(((torch::THPVoidTensor*)obj)->cdata));
  REGISTER_THC_DESCRIPTOR(LongTensor);
  REGISTER_THC_DESCRIPTOR(IntTensor);
  REGISTER_THC_DESCRIPTOR(ShortTensor);
  REGISTER_THC_DESCRIPTOR(CharTensor);
  REGISTER_THC_DESCRIPTOR(ByteTensor);
#undef REGISTER_THC_DESCRIPTOR
#endif
  throw std::runtime_error(std::string("don't know how to create a THDTensorDesciptor for "
      "type ") + std::string(THPUtils_typename(obj)));
}

static THDRequest* _unpackRequest(PyObject *obj)
{
  return static_cast<THDRequest*>(THPWrapper_get(obj));
}

static THDReduceOp _getReduceOp(PyObject *obj)
{
  auto it = obj2reduceop.find(obj);
  if (it == obj2reduceop.end()) {
    throw std::runtime_error("op should be a constant from "
        "torch.distributed.reduce_op");
  }
  return it->second;
}

static THDGroup _getGroup(PyObject *obj)
{
  auto it = obj2group.find(obj);
  if (it == obj2group.end()) {
    if (!THPUtils_checkLong(obj))
      throw std::runtime_error("group should be an int or one of the values "
          "from torch.distributed.group");
    return THPUtils_unpackLong(obj);
  }
  return it->second;
}

PyObject* THDPModule_isend(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "isend", 1, "(tensor input, int dst_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  return THPWrapper_New(THDIsend(desc, dst_rank), (void(*)(void*))THDRequest_free);
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_irecv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "irecv", 1, "(tensor output, int src_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  return THPWrapper_New(THDIrecv(desc, src_rank), (void(*)(void*))THDRequest_free);
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_send(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "send", 1, "(tensor input, int dst_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDSend(desc, dst_rank);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_recvAnySource(PyObject *_unused, PyObject *_tensor)
{
  HANDLE_TH_ERRORS
  if (!THPModule_isTensor(_tensor)) {
    THPUtils_invalidArguments(_tensor, NULL, "recv", 1, "(tensor output)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(_tensor);
  THDRecvAnySource(desc);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_recv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "recv", 1, "(tensor output, int src_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDRecv(desc, src_rank);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_allReduce(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0))) {
    THPUtils_invalidArguments(args, NULL, "all_reduce", 1, "(tensor in_out, reduce_op op, group gr)");
    return NULL;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  THDReduceOp op = _getReduceOp(PyTuple_GET_ITEM(args, 1));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  THDAllReduce(desc, op, group);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_reduce(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 4 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "reduce", 1,
        "(tensor reduced, int dst_rank, reduce_op op, group gr)");
    return NULL;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 3));
  THDReduceOp op = _getReduceOp(PyTuple_GET_ITEM(args, 2));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDReduce(desc, op, dst_rank, group);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_broadcast(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "broadcast", 1,
        "(tensor src_dst, int src_rank, group gr)");
    return NULL;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDBroadcast(desc, src_rank, group);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_allGather(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* sequence = PyTuple_GET_ITEM(args, 0);
  Py_ssize_t tmp_length;
  std::size_t length;
  std::vector<THDPTensorDesc> descriptors;
  std::vector<THDTensorDescriptor*> raw_descriptors;

  if (PyTuple_GET_SIZE(args) != 3 || !PySequence_Check(sequence) ||
        !THPModule_isTensor(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  tmp_length = PySequence_Length(sequence);
  THPUtils_assert(tmp_length >= 0, "couldn't obtain the length of %s",
      THPUtils_typename(sequence));

  length = static_cast<std::size_t>(tmp_length);
  descriptors.reserve(length);
  for (std::size_t i = 0; i < length; ++i) {
    if (!THPModule_isTensor(PySequence_ITEM(sequence, i)))
      goto invalid_arguments;

    descriptors.push_back(
      THDPTensorDesc(_makeDescriptor(PySequence_ITEM(sequence, i)))
    );
    raw_descriptors.push_back(descriptors.back());
  }

  THDAllGather(
    raw_descriptors.data(), length,
    THDPTensorDesc(_makeDescriptor(PyTuple_GET_ITEM(args, 1))),
    _getGroup(PyTuple_GET_ITEM(args, 2))
  );
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, NULL, "allGather", 1,
      "(list[tensor] output, tensor input, group gr)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_gatherSend(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0))) {
    THPUtils_invalidArguments(args, NULL, "gatherSend", 1,
        "(tensor input, int dst_rank, group gr)");
    return NULL;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDGatherSend(desc, dst_rank, group);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_gatherRecv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* sequence = PyTuple_GET_ITEM(args, 0);
  Py_ssize_t tmp_length;
  std::size_t length;
  std::vector<THDPTensorDesc> descriptors;
  std::vector<THDTensorDescriptor*> raw_descriptors;

  if (PyTuple_GET_SIZE(args) != 3 || !PySequence_Check(sequence) ||
        !THPModule_isTensor(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  tmp_length = PySequence_Length(sequence);
  THPUtils_assert(tmp_length >= 0, "couldn't obtain the length of %s",
      THPUtils_typename(sequence));

  length = static_cast<std::size_t>(tmp_length);
  descriptors.reserve(length);
  for (std::size_t i = 0; i < length; ++i) {
    if (!THPModule_isTensor(PySequence_ITEM(sequence, i)))
      goto invalid_arguments;

    descriptors.push_back(
      THDPTensorDesc(_makeDescriptor(PySequence_ITEM(sequence, i)))
    );
    raw_descriptors.push_back(descriptors.back());
  }

  THDGatherRecv(
    raw_descriptors.data(), length,
    THDPTensorDesc(_makeDescriptor(PyTuple_GET_ITEM(args, 1))),
    _getGroup(PyTuple_GET_ITEM(args, 2))
  );
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, NULL, "gatherRecv", 1,
      "(list[tensor] output, tensor input, group gr)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_scatterSend(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* sequence = PyTuple_GET_ITEM(args, 0);
  Py_ssize_t tmp_length;
  std::size_t length;
  std::vector<THDPTensorDesc> descriptors;
  std::vector<THDTensorDescriptor*> raw_descriptors;

  if (PyTuple_GET_SIZE(args) != 3 || !PySequence_Check(sequence) ||
        !THPModule_isTensor(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  tmp_length = PySequence_Length(sequence);
  THPUtils_assert(tmp_length >= 0, "couldn't obtain the length of %s",
      THPUtils_typename(sequence));

  length = static_cast<std::size_t>(tmp_length);
  descriptors.reserve(length);
  for (std::size_t i = 0; i < length; ++i) {
    if (!THPModule_isTensor(PySequence_ITEM(sequence, i)))
      goto invalid_arguments;

    descriptors.push_back(
      THDPTensorDesc(_makeDescriptor(PySequence_ITEM(sequence, i)))
    );
    raw_descriptors.push_back(descriptors.back());
  }

  THDScatterSend(
    raw_descriptors.data(), length,
    THDPTensorDesc(_makeDescriptor(PyTuple_GET_ITEM(args, 1))),
    _getGroup(PyTuple_GET_ITEM(args, 2))
  );
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, NULL, "scatterSend", 1,
      "(list[tensor] input, tensor output, group gr)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_scatterRecv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, NULL, "scatterRecv", 1,
        "(tensor output, int src_rank, group gr)");
    return NULL;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDScatterRecv(desc, src_rank, group);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_barrier(PyObject *_unused, PyObject *_group)
{
  HANDLE_TH_ERRORS
  THDBarrier(_getGroup(_group));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_newGroup(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* sequence = PyTuple_GET_ITEM(args, 0);
  Py_ssize_t tmp_length;
  std::size_t length;
  std::vector<int> ranks;

  if (PyTuple_GET_SIZE(args) != 1 || !PySequence_Check(sequence))
    goto invalid_arguments;

  tmp_length = PySequence_Length(sequence);
  THPUtils_assert(tmp_length >= 0, "couldn't obtain the length of %s",
      THPUtils_typename(sequence));

  length = static_cast<std::size_t>(tmp_length);
  ranks.reserve(length);
  for (std::size_t i = 0; i < length; ++i) {
    if (!THPUtils_checkLong(PySequence_ITEM(sequence, i)))
      goto invalid_arguments;

    ranks.push_back(THPUtils_unpackLong(PySequence_ITEM(sequence, i)));
    for (std::size_t j = 0; j < i; ++j)
      THPUtils_assert(ranks[i] != ranks[j], "ranks should be unique");
  }

  return PyInt_FromLong(THDNewGroup(ranks.data(), length));

invalid_arguments:
  THPUtils_invalidArguments(args, NULL, "newGroup", 1, "(list[int] ranks)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_requestIsCompleted(PyObject *_unused, PyObject *_req)
{
  HANDLE_TH_ERRORS
  if (!THPWrapper_check(_req)) {
    THPUtils_invalidArguments(_req, NULL, "requestIsCompleted", 1, "(request req)");
    return NULL;
  }

  return PyBool_FromLong(THDRequest_isCompleted(_unpackRequest(_req)));
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_requestWait(PyObject *_unused, PyObject *_req)
{
  HANDLE_TH_ERRORS
  if (!THPWrapper_check(_req)) {
    THPUtils_invalidArguments(_req, NULL, "requestWait", 1, "(request req)");
    return NULL;
  }

  THDRequest_wait(_unpackRequest(_req));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_initExtension(PyObject *_unused, PyObject *args) {
  if (PyTuple_GET_SIZE(args) != 3) {
    THPUtils_invalidArguments(args, NULL, "initExtension", 1, "(bool is_master_worker, reduce_op obj, group obj)");
    return NULL;
  }

  PyObject* is_master_worker_obj = PyTuple_GET_ITEM(args, 0);
  PyObject* reduce_op_obj = PyTuple_GET_ITEM(args, 1);
  PyObject* group_obj = PyTuple_GET_ITEM(args, 2);

  THPUtils_assert(PyBool_Check(is_master_worker_obj), "first argument should be a bool");
  bool is_master_worker = is_master_worker_obj == Py_True;

  THPObjectPtr reduce_op;
#define REGISTER_REDUCE_OP(NAME)                                               \
  reduce_op = PyObject_GetAttrString(reduce_op_obj, #NAME);                    \
  THPUtils_assert(reduce_op, "Missing object for reduce op " #NAME);           \
  obj2reduceop.emplace(reduce_op.get(), THDReduce##NAME);
  REGISTER_REDUCE_OP(SUM);
  REGISTER_REDUCE_OP(PRODUCT);
  REGISTER_REDUCE_OP(MIN);
  REGISTER_REDUCE_OP(MAX);
#undef REGISTER_REDUCE_OP

  THPObjectPtr group;
#define REGISTER_GROUP(NAME)                                           \
  group = PyObject_GetAttrString(group_obj, #NAME);                    \
  THPUtils_assert(group, "Missing object for group " #NAME);           \
  obj2group.emplace(group.get(), THDGroup##NAME);
  REGISTER_GROUP(WORLD);
#undef REGISTER_GROUP

  if (is_master_worker) {
    PyObject *module = PyImport_ImportModule("torch.distributed");
    THPUtils_assert(module, "class loader couldn't access torch.distributed module");
    PyObject* module_dict = PyModule_GetDict(module);
    if (!THDPModule_loadClasses(module_dict)) return NULL;
  }
  Py_RETURN_TRUE;
}

static struct PyMethodDef _THDPModule_methods[] = {
  {"_dist_init_extension", (PyCFunction)THDPModule_initExtension, METH_VARARGS, NULL},
  {"_dist_init_process_group", (PyCFunction)THDPModule_initProcessGroup, METH_VARARGS, NULL},
  {"_dist_init_master_worker", (PyCFunction)THDPModule_initMasterWorker, METH_O, NULL},
  {"_dist_get_rank", (PyCFunction)THDPModule_getRank, METH_NOARGS, NULL},
  {"_dist_get_num_processes", (PyCFunction)THDPModule_getNumProcesses, METH_NOARGS, NULL},
  {"_dist_isend", (PyCFunction)THDPModule_isend, METH_VARARGS, NULL},
  {"_dist_irecv", (PyCFunction)THDPModule_irecv, METH_VARARGS, NULL},
  {"_dist_send", (PyCFunction)THDPModule_send, METH_VARARGS, NULL},
  {"_dist_recv_any_source", (PyCFunction)THDPModule_recvAnySource, METH_O, NULL},
  {"_dist_recv", (PyCFunction)THDPModule_recv, METH_VARARGS, NULL},
  {"_dist_all_reduce", (PyCFunction)THDPModule_allReduce, METH_VARARGS, NULL},
  {"_dist_reduce", (PyCFunction)THDPModule_reduce, METH_VARARGS, NULL},
  {"_dist_broadcast", (PyCFunction)THDPModule_broadcast, METH_VARARGS, NULL},
  {"_dist_all_gather", (PyCFunction)THDPModule_allGather, METH_VARARGS, NULL},
  {"_dist_gather_send", (PyCFunction)THDPModule_gatherSend, METH_VARARGS, NULL},
  {"_dist_gather_recv", (PyCFunction)THDPModule_gatherRecv, METH_VARARGS, NULL},
  {"_dist_scatter_send", (PyCFunction)THDPModule_scatterSend, METH_VARARGS, NULL},
  {"_dist_scatter_recv", (PyCFunction)THDPModule_scatterRecv, METH_VARARGS, NULL},
  {"_dist_barrier", (PyCFunction)THDPModule_barrier, METH_O, NULL},
  {"_dist_new_group", (PyCFunction)THDPModule_newGroup, METH_VARARGS, NULL},
  {"_dist_request_is_completed", (PyCFunction)THDPModule_requestIsCompleted, METH_O, NULL},
  {"_dist_request_wait", (PyCFunction)THDPModule_requestWait, METH_O, NULL},
  {NULL}
};

PyMethodDef* THDPModule_methods() {
  return _THDPModule_methods;
}
