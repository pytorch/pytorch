#include <Python.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include "THDP.h"

static std::unordered_map<std::string, THDChannelType> name2channel_type = {
    {"mpi", THDChannelMPI},
    {"tcp", THDChannelTCP},
};

static std::unordered_map<PyObject*, THDReduceOp> obj2reduceop;
static std::unordered_map<PyObject*, THDGroup> obj2group;

static THPObjectPtr _ensureBytes(PyObject *obj)
{
#if PY_MAJOR_VERSION == 2
  if (PyString_Check(obj)) {
#elif PY_MAJOR_VERSION == 3
  if (PyBytes_Check(obj)) {
#endif
    Py_INCREF(obj);
    return obj;
  }
  if (PyUnicode_Check(obj)) {
    return PyUnicode_AsASCIIString(obj);
  }
  return NULL;
}

PyObject* THDPModule_initProcessGroup(PyObject *_unused, PyObject *_backend)
{
  HANDLE_TH_ERRORS
  THPObjectPtr backend_bytes = _ensureBytes(_backend);
  THPUtils_assert(backend_bytes, "backend argument has to be a string/bytes "
      "object, but got %s", THPUtils_typename(_backend));
  char *backend_name = THPUtils_bytesAsString(backend_bytes.get());
  THDChannelType channel_type = name2channel_type.at(backend_name);
  THPUtils_assert(THDProcessGroupInit(channel_type), "failed to initialize "
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
  throw std::runtime_error(std::string("don't know how to create a THDTensorDesciptor for "
      "type ") + std::string(THPUtils_typename(obj)));
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

PyObject* THDPModule_send(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, "send", 1, "(tensor input, int dst_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDSend(desc, dst_rank);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_recv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, "recv", 1, "(tensor output, int src_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDReceive(desc, src_rank);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_allReduce(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0))) {
    THPUtils_invalidArguments(args, "all_reduce", 1, "(tensor in_out, reduce_op op, group gr)");
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
    THPUtils_invalidArguments(args, "reduce", 1,
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
    THPUtils_invalidArguments(args, "broadcast", 1,
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

PyObject* THDPModule_newGroup(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  PyObject* sequence = PyTuple_GET_ITEM(args, 0);
  Py_ssize_t tmp_length;
  std::vector<int> ranks;

  if (PyTuple_GET_SIZE(args) != 1 || !PySequence_Check(sequence))
    goto invalid_arguments;

  tmp_length = PySequence_Length(sequence);
  THPUtils_assert(tmp_length >= 0, "couldn't obtain the length of %s",
      THPUtils_typename(sequence));

  ranks.reserve(static_cast<std::size_t>(tmp_length));
  for (std::size_t i = 0; i < ranks.capacity(); ++i) {
    if (!THPUtils_checkLong(PySequence_ITEM(sequence, i)))
      goto invalid_arguments;

    ranks.push_back(THPUtils_unpackLong(PySequence_ITEM(sequence, i)));
    for (std::size_t j = 0; j < i; ++j)
      THPUtils_assert(ranks[i] != ranks[j], "ranks should be unique");
  }

  return PyInt_FromLong(THDNewGroup(ranks.data(), ranks.size()));

invalid_arguments:
  THPUtils_invalidArguments(args, "newGroup", 1, "(list[int] ranks)");
  return NULL;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_initExtension(PyObject *_unused, PyObject *args) {
  if (PyTuple_GET_SIZE(args) != 2) {
    THPUtils_invalidArguments(args, "initExtension", 1, "(reduce_op obj, group obj)");
    return NULL;
  }

  PyObject* reduce_op_obj = PyTuple_GET_ITEM(args, 0);
  PyObject* group_obj = PyTuple_GET_ITEM(args, 1);

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
  Py_RETURN_TRUE;
}

static struct PyMethodDef _THDPModule_methods[] = {
  {"_dist_init_extension", (PyCFunction)THDPModule_initExtension, METH_VARARGS, NULL},
  {"_dist_init_process_group", (PyCFunction)THDPModule_initProcessGroup, METH_O, NULL},
  {"_dist_get_rank", (PyCFunction)THDPModule_getRank, METH_NOARGS, NULL},
  {"_dist_get_num_processes", (PyCFunction)THDPModule_getNumProcesses, METH_NOARGS, NULL},
  {"_dist_send", (PyCFunction)THDPModule_send, METH_VARARGS, NULL},
  {"_dist_recv", (PyCFunction)THDPModule_recv, METH_VARARGS, NULL},
  {"_dist_all_reduce", (PyCFunction)THDPModule_allReduce, METH_VARARGS, NULL},
  {"_dist_reduce", (PyCFunction)THDPModule_reduce, METH_VARARGS, NULL},
  {"_dist_broadcast", (PyCFunction)THDPModule_broadcast, METH_VARARGS, NULL},
  {"_dist_new_group", (PyCFunction)THDPModule_newGroup, METH_VARARGS, NULL},
  {NULL}
};

PyMethodDef* THDPModule_methods() {
  return _THDPModule_methods;
}
