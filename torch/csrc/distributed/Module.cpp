#include <Python.h>

#include <unordered_map>

#include "THDP.h"

static std::unordered_map<std::string, THDChannelType> name2channel_type = {
    {"mpi", THDChannelMPI},
    {"tcp", THDChannelTCP},
};

static std::unordered_map<PyObject*, THDReduceOp> obj2reduceop;

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
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0))) {
    THPUtils_invalidArguments(args, "all_reduce", 1, "(tensor in_out, reduce_op op)");
    return NULL;
  }

  THDReduceOp op = _getReduceOp(PyTuple_GET_ITEM(args, 1));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  THDAllReduce(desc, op);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_reduce(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, "reduce", 1,
        "(tensor reduced, int dst_rank, reduce_op op)");
    return NULL;
  }

  THDReduceOp op = _getReduceOp(PyTuple_GET_ITEM(args, 2));
  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDReduce(desc, op, dst_rank);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_broadcast(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPModule_isTensor(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, "broadcast", 1, "(tensor src_dst, int src_rank)");
    return NULL;
  }

  THDPTensorDesc desc = _makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDBroadcast(desc, src_rank);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_initExtension(PyObject *_unused, PyObject *reduce_op_obj) {
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
  Py_RETURN_TRUE;
}

static struct PyMethodDef _THDPModule_methods[] = {
  {"_dist_init_extension", (PyCFunction)THDPModule_initExtension, METH_O, NULL},
  {"_dist_init_process_group", (PyCFunction)THDPModule_initProcessGroup, METH_O, NULL},
  {"_dist_get_rank", (PyCFunction)THDPModule_getRank, METH_NOARGS, NULL},
  {"_dist_get_num_processes", (PyCFunction)THDPModule_getNumProcesses, METH_NOARGS, NULL},
  {"_dist_send", (PyCFunction)THDPModule_send, METH_VARARGS, NULL},
  {"_dist_recv", (PyCFunction)THDPModule_recv, METH_VARARGS, NULL},
  {"_dist_all_reduce", (PyCFunction)THDPModule_allReduce, METH_VARARGS, NULL},
  {"_dist_reduce", (PyCFunction)THDPModule_reduce, METH_VARARGS, NULL},
  {"_dist_broadcast", (PyCFunction)THDPModule_broadcast, METH_VARARGS, NULL},
  {NULL}
};

PyMethodDef* THDPModule_methods() {
  return _THDPModule_methods;
}
