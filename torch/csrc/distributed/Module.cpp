#include <torch/csrc/python_headers.h>

#include <memory>
#include <unordered_map>
#include <vector>

#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/distributed/THDP.h>
#include <torch/csrc/PythonTypes.h>
#include <torch/csrc/autograd/python_variable.h>

#ifdef USE_CUDA
#include <torch/csrc/cuda/Stream.h>
#endif


static std::unordered_map<std::string, THDChannelType> name2channel_type = {
    {"mpi", THDChannelMPI},
    {"tcp", THDChannelTCP},
    {"gloo", THDChannelGloo},
    {"nccl", THDChannelNccl},
};

static std::unordered_map<PyObject*, THDReduceOp> obj2reduceop;
static std::unordered_map<PyObject*, THDGroup> obj2group;

#ifdef USE_CUDA
extern THCState* state;
#endif

PyObject* THDPModule_initProcessGroup(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 5 || !THPUtils_checkString(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkString(PyTuple_GET_ITEM(args, 1)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 2)) ||
        !THPUtils_checkString(PyTuple_GET_ITEM(args, 3)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 4))) {
    THPUtils_invalidArguments(args, nullptr, "init_process_group", 1, "(string backend, string init_method, int world_size, string group_name, int rank)");
    return nullptr;
  }

  std::string backend_name = THPUtils_unpackString(PyTuple_GET_ITEM(args, 0));
  std::string init_method = THPUtils_unpackString(PyTuple_GET_ITEM(args, 1));
  int world_size = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 2));
  std::string group_name = THPUtils_unpackString(PyTuple_GET_ITEM(args, 3));
  int rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 4));

  THDChannelType channel_type = name2channel_type.at(backend_name);
  {
    AutoNoGIL nogil;
    THDProcessGroupInit(channel_type, init_method, world_size, group_name, rank);
  }
#ifdef USE_CUDA
  THDSetCudaStatePtr(&state);
#endif
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_destroyProcessGroup(PyObject *_unused) {
  HANDLE_TH_ERRORS
  {
    AutoNoGIL nogil;
    THDProcessGroupDestroy();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifdef USE_CUDA
PyObject* THDPModule_registerStream(PyObject *_unused, PyObject *_stream)
{
  HANDLE_TH_ERRORS
  THPUtils_assert(THCPStream_Check(_stream), "_register_stream expects a "
      "torch.cuda.Stream object");
  THCPStream *stream = (THCPStream*)_stream;
  THDRegisterCudaStream(stream->cuda_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}
#endif

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

#ifdef USE_CUDA
extern PyObject* THCPDoubleTensorClass;
extern PyObject* THCPFloatTensorClass;
extern PyObject* THCPHalfTensorClass;
extern PyObject* THCPLongTensorClass;
extern PyObject* THCPIntTensorClass;
extern PyObject* THCPShortTensorClass;
extern PyObject* THCPCharTensorClass;
extern PyObject* THCPByteTensorClass;
#endif

THDTensorDescriptor THDPModule_makeDescriptor(PyObject *obj) {
  auto var = (THPVariable*)obj;
  return var->cdata.tensor_data();
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
        "torch.distributed.deprecated.reduce_op");
  }
  return it->second;
}

static THDGroup _getGroup(PyObject *obj)
{
  auto it = obj2group.find(obj);
  if (it == obj2group.end()) {
    if (!THPUtils_checkLong(obj))
      throw std::runtime_error("group should be an int or one of the values "
          "from torch.distributed.deprecated.group");
    return THPUtils_unpackLong(obj);
  }
  return it->second;
}

PyObject* THDPModule_clearGroupCache(PyObject *_unused, PyObject *args) {
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 1) {
    THPUtils_invalidArguments(args, nullptr, "clear_group_cache", 1, "(group gr)");
    return nullptr;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 0));

  {
    AutoNoGIL nogil;
    THDClearGroupCache(group);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_isend(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "isend", 1, "(tensor input, int dst_rank)");
    return nullptr;
  }

  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDRequest* req;
  {
    AutoNoGIL guard;
    req = THDIsend(desc, dst_rank);
  }
  return THPWrapper_New(req, (void(*)(void*))THDRequest_free);
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_irecv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "irecv", 1, "(tensor output, int src_rank)");
    return nullptr;
  }

  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  THDRequest* req;
  {
    AutoNoGIL guard;
    req = THDIrecv(desc, src_rank);
  }
  return THPWrapper_New(req, (void(*)(void*))THDRequest_free);
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_send(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "send", 1, "(tensor input, int dst_rank)");
    return nullptr;
  }

  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDSend(desc, dst_rank);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_recvAnySource(PyObject *_unused, PyObject *_tensor)
{
  HANDLE_TH_ERRORS
  if (!THPVariable_Check(_tensor)) {
    THPUtils_invalidArguments(_tensor, nullptr, "recv", 1, "(tensor output)");
    return nullptr;
  }

  auto desc = THDPModule_makeDescriptor(_tensor);
  int sender;
  {
    AutoNoGIL guard;
    sender = THDRecvAnySource(desc);
  }
  return PyInt_FromLong(sender);
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_recv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 2 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "recv", 1, "(tensor output, int src_rank)");
    return nullptr;
  }

  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDRecv(desc, src_rank);
  }
  // Return sender rank
  Py_INCREF(PyTuple_GET_ITEM(args, 1));
  return PyTuple_GET_ITEM(args, 1);
  END_HANDLE_TH_ERRORS
}


PyObject* THDPModule_allReduceMultiGPU(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  std::vector<at::Tensor> descriptors;
  size_t length;
  THDGroup group;
  THDReduceOp op;
  THPObjectPtr sequence;

  if (PyTuple_GET_SIZE(args) != 3) {
    goto invalid_arguments;
  }

  if (!PySequence_Check(PyTuple_GET_ITEM(args, 0))) {
    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  descriptors.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence.get(), i))) {
      goto invalid_arguments;
    }

    descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence.get(), i))
    );
  }

  group = _getGroup(PyTuple_GET_ITEM(args, 2));
  op = _getReduceOp(PyTuple_GET_ITEM(args, 1));

  {
    AutoNoGIL guard;
    THDAllReduceMultiGPU(descriptors.data(), length, op, group);
  }
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "all_reduce_multigpu", 1,
                            "(list[tensor] in_out, reduce_op op, group gr)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}


PyObject* THDPModule_reduceMultiGPU(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence;
  size_t length;
  std::vector<at::Tensor> descriptors;
  THDGroup group;
  THDReduceOp op;
  int dst_rank;

  if (PyTuple_GET_SIZE(args) != 4) {
    goto invalid_arguments;
  }

  if (!PySequence_Check(PyTuple_GET_ITEM(args, 0)) ||
      !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  descriptors.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence.get(), i))) {
      goto invalid_arguments;
    }

    descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence.get(), i))
    );
  }

  group = _getGroup(PyTuple_GET_ITEM(args, 3));
  op = _getReduceOp(PyTuple_GET_ITEM(args, 2));
  dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));

  {
    AutoNoGIL guard;
    THDReduceMultiGPU(descriptors.data(), length, op, dst_rank, group);
  }
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "reduce_multigpu", 1,
                            "(list[tensor] in_out, int dst_rank, "
                            "reduce_op op, group gr)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}


PyObject* THDPModule_broadcastMultiGPU(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence;
  size_t length;
  std::vector<at::Tensor> descriptors;
  THDGroup group;
  int src_rank;

  if (PyTuple_GET_SIZE(args) != 3) {
    goto invalid_arguments;
  }

  if (!PySequence_Check(PyTuple_GET_ITEM(args, 0)) ||
      !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  descriptors.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence.get(), i))) {
      goto invalid_arguments;
    }

    descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence.get(), i))
    );
  }

  group = _getGroup(PyTuple_GET_ITEM(args, 2));
  src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));

  {
    AutoNoGIL guard;
    THDBroadcastMultiGPU(descriptors.data(), length, src_rank, group);
  }
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "broadcast_multigpu", 1,
                            "(list[tensor] in_out, int src_rank, group gr)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}


PyObject* THDPModule_allGatherMultiGPU(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence_one;
  THPObjectPtr sequence_two;

  size_t length_one;
  size_t length_two;

  std::vector<at::Tensor> output_descriptors;
  std::vector<at::Tensor> input_descriptors;

  THDGroup group;

  if (PyTuple_GET_SIZE(args) != 3) {
    goto invalid_arguments;
  }

  if (!PySequence_Check(PyTuple_GET_ITEM(args, 0)) ||
      !PySequence_Check(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  sequence_one = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                                              "expected a sequence"));
  sequence_two = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 1),
                                              "expected a sequence"));

  if (!sequence_one.get() || !sequence_two.get()) {
    goto invalid_arguments;
  }

  length_one = static_cast<size_t>(
      PySequence_Fast_GET_SIZE(sequence_one.get()));

  length_two = static_cast<size_t>(
      PySequence_Fast_GET_SIZE(sequence_two.get()));

  if (length_one != length_two) {
    goto invalid_arguments;
  }

  output_descriptors.reserve(length_one);
  input_descriptors.reserve(length_two);

  // Get the input list
  for (size_t i = 0; i < length_two; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence_two.get(), i)) ||
        !THPVariable_Check(PySequence_Fast_GET_ITEM(sequence_one.get(), i))) {
      goto invalid_arguments;
    }

    input_descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence_two.get(), i))
    );

    output_descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence_one.get(), i))
    );
  }

  group = _getGroup(PyTuple_GET_ITEM(args, 2));

  {
    AutoNoGIL guard;
    THDAllGatherMultiGPU(output_descriptors.data(),
                         length_one,
                         input_descriptors.data(),
                         length_two,
                         group);
  }

  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "all_gather_multigpu", 1,
      "(list[list[tensor]] output, list[tensor] input, group gr)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}


PyObject* THDPModule_allReduce(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0))) {
    THPUtils_invalidArguments(args, nullptr, "all_reduce", 1, "(tensor in_out, reduce_op op, group gr)");
    return nullptr;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  THDReduceOp op = _getReduceOp(PyTuple_GET_ITEM(args, 1));
  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  {
    AutoNoGIL guard;
    THDAllReduce(desc, op, group);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_reduce(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 4 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "reduce", 1,
        "(tensor reduced, int dst_rank, reduce_op op, group gr)");
    return nullptr;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 3));
  THDReduceOp op = _getReduceOp(PyTuple_GET_ITEM(args, 2));
  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDReduce(desc, op, dst_rank, group);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_broadcast(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "broadcast", 1,
        "(tensor src_dst, int src_rank, group gr)");
    return nullptr;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDBroadcast(desc, src_rank, group);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_allGather(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence;
  size_t length;
  std::vector<at::Tensor> descriptors;
  THDGroup group;
  at::Tensor desc;

  if (PyTuple_GET_SIZE(args) != 3 ||
      !PySequence_Check(PyTuple_GET_ITEM(args, 0)) ||
      !THPVariable_Check(PyTuple_GET_ITEM(args, 1))) {

    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  descriptors.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence.get(), i)))
      goto invalid_arguments;

    descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence.get(), i))
    );
  }

  group = _getGroup(PyTuple_GET_ITEM(args, 2));
  desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDAllGather(descriptors.data(), length, desc, group);
  }
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "allGather", 1,
      "(list[tensor] output, tensor input, group gr)");
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_gatherSend(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0))) {
    THPUtils_invalidArguments(args, nullptr, "gatherSend", 1,
        "(tensor input, int dst_rank, group gr)");
    return nullptr;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int dst_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDGatherSend(desc, dst_rank, group);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_gatherRecv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence;
  size_t length;
  std::vector<at::Tensor> descriptors;
  THDGroup group;
  at::Tensor desc;

  if (PyTuple_GET_SIZE(args) != 3 ||
      !PySequence_Check(PyTuple_GET_ITEM(args, 0)) ||
      !THPVariable_Check(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  descriptors.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence.get(), i)))
      goto invalid_arguments;

    descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence.get(), i))
    );
  }

  desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 1));
  group = _getGroup(PyTuple_GET_ITEM(args, 2));
  {
    AutoNoGIL guard;
    THDGatherRecv(descriptors.data(), length, desc, group);
  }
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "gatherRecv", 1,
      "(list[tensor] output, tensor input, group gr)");
  return nullptr;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_scatterSend(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence;
  size_t length;
  std::vector<at::Tensor> descriptors;
  THDGroup group;
  at::Tensor desc;

  if (PyTuple_GET_SIZE(args) != 3 ||
      !PySequence_Check(PyTuple_GET_ITEM(args, 0)) ||
      !THPVariable_Check(PyTuple_GET_ITEM(args, 1))) {
    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  descriptors.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPVariable_Check(PySequence_Fast_GET_ITEM(sequence.get(), i)))
      goto invalid_arguments;

    descriptors.push_back(
      THDPModule_makeDescriptor(PySequence_Fast_GET_ITEM(sequence.get(), i))
    );
  }

  desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 1));
  group = _getGroup(PyTuple_GET_ITEM(args, 2));
  {
    AutoNoGIL guard;
    THDScatterSend(descriptors.data(), length, desc, group);
  }
  Py_RETURN_NONE;

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "scatterSend", 1,
      "(list[tensor] input, tensor output, group gr)");
  return nullptr;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_scatterRecv(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  if (PyTuple_GET_SIZE(args) != 3 || !THPVariable_Check(PyTuple_GET_ITEM(args, 0)) ||
        !THPUtils_checkLong(PyTuple_GET_ITEM(args, 1))) {
    THPUtils_invalidArguments(args, nullptr, "scatterRecv", 1,
        "(tensor output, int src_rank, group gr)");
    return nullptr;
  }

  THDGroup group = _getGroup(PyTuple_GET_ITEM(args, 2));
  auto desc = THDPModule_makeDescriptor(PyTuple_GET_ITEM(args, 0));
  int src_rank = THPUtils_unpackLong(PyTuple_GET_ITEM(args, 1));
  {
    AutoNoGIL guard;
    THDScatterRecv(desc, src_rank, group);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_barrier(PyObject *_unused, PyObject *_group)
{
  HANDLE_TH_ERRORS
  {
    AutoNoGIL guard;
    THDBarrier(_getGroup(_group));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_newGroup(PyObject *_unused, PyObject *args)
{
  HANDLE_TH_ERRORS
  THPObjectPtr sequence;
  size_t length;
  std::vector<int> ranks;

  if (PyTuple_GET_SIZE(args) != 1 ||
      !PySequence_Check(PyTuple_GET_ITEM(args, 0))) {
    goto invalid_arguments;
  }

  sequence = THPObjectPtr(PySequence_Fast(PyTuple_GET_ITEM(args, 0),
                                          "expected a sequence"));
  if (!sequence.get()) {
    goto invalid_arguments;
  }

  length = static_cast<size_t>(PySequence_Fast_GET_SIZE(sequence.get()));

  ranks.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (!THPUtils_checkLong(PySequence_Fast_GET_ITEM(sequence.get(), i)))
      goto invalid_arguments;

    ranks.push_back(THPUtils_unpackLong(
          PySequence_Fast_GET_ITEM(sequence.get(), i)));

    for (size_t j = 0; j < i; ++j)
      THPUtils_assert(ranks[i] != ranks[j], "ranks should be unique");
  }

  THDGroup group;
  {
    AutoNoGIL guard;
    group = THDNewGroup(ranks.data(), length);
  }
  return PyInt_FromLong(group);

invalid_arguments:
  THPUtils_invalidArguments(args, nullptr, "newGroup", 1, "(list[int] ranks)");
  return nullptr;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_requestIsCompleted(PyObject *_unused, PyObject *_req)
{
  HANDLE_TH_ERRORS
  if (!THPWrapper_check(_req)) {
    THPUtils_invalidArguments(_req, nullptr, "requestIsCompleted", 1, "(request req)");
    return nullptr;
  }

  return PyBool_FromLong(THDRequest_isCompleted(_unpackRequest(_req)));
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_requestWait(PyObject *_unused, PyObject *_req)
{
  HANDLE_TH_ERRORS
  if (!THPWrapper_check(_req)) {
    THPUtils_invalidArguments(_req, nullptr, "requestWait", 1, "(request req)");
    return nullptr;
  }

  {
    AutoNoGIL guard;
    THDRequest_wait(_unpackRequest(_req));
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THDPModule_initExtension(PyObject *_unused, PyObject *args) {
  if (PyTuple_GET_SIZE(args) != 3) {
    THPUtils_invalidArguments(args, nullptr, "initExtension", 1, "(bool is_master_worker, reduce_op obj, group obj)");
    return nullptr;
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
    throw std::runtime_error("THD master_worker no longer supported");
  }
  Py_RETURN_TRUE;
}

static struct PyMethodDef _THDPModule_methods[] = {
  {"_dist_init_extension", (PyCFunction)THDPModule_initExtension, METH_VARARGS, nullptr},
  {"_dist_init_process_group", (PyCFunction)THDPModule_initProcessGroup, METH_VARARGS, nullptr},
  {"_dist_destroy_process_group", (PyCFunction)THDPModule_destroyProcessGroup, METH_NOARGS, nullptr},
  {"_dist_clear_group_cache", (PyCFunction)THDPModule_clearGroupCache, METH_VARARGS, nullptr},
#ifdef USE_CUDA
  {"_dist_register_stream", (PyCFunction)THDPModule_registerStream, METH_O, nullptr},
#endif
  {"_dist_get_rank", (PyCFunction)THDPModule_getRank, METH_NOARGS, nullptr},
  {"_dist_get_num_processes", (PyCFunction)THDPModule_getNumProcesses, METH_NOARGS, nullptr},
  {"_dist_isend", (PyCFunction)THDPModule_isend, METH_VARARGS, nullptr},
  {"_dist_irecv", (PyCFunction)THDPModule_irecv, METH_VARARGS, nullptr},
  {"_dist_send", (PyCFunction)THDPModule_send, METH_VARARGS, nullptr},
  {"_dist_recv_any_source", (PyCFunction)THDPModule_recvAnySource, METH_O, nullptr},
  {"_dist_recv", (PyCFunction)THDPModule_recv, METH_VARARGS, nullptr},
  {"_dist_all_reduce", (PyCFunction)THDPModule_allReduce, METH_VARARGS, nullptr},
  {"_dist_all_reduce_multigpu", (PyCFunction)THDPModule_allReduceMultiGPU, METH_VARARGS, nullptr},
  {"_dist_reduce", (PyCFunction)THDPModule_reduce, METH_VARARGS, nullptr},
  {"_dist_reduce_multigpu", (PyCFunction)THDPModule_reduceMultiGPU, METH_VARARGS, nullptr},
  {"_dist_broadcast", (PyCFunction)THDPModule_broadcast, METH_VARARGS, nullptr},
  {"_dist_broadcast_multigpu", (PyCFunction)THDPModule_broadcastMultiGPU, METH_VARARGS, nullptr},
  {"_dist_all_gather", (PyCFunction)THDPModule_allGather, METH_VARARGS, nullptr},
  {"_dist_all_gather_multigpu", (PyCFunction)THDPModule_allGatherMultiGPU, METH_VARARGS, nullptr},
  {"_dist_gather_send", (PyCFunction)THDPModule_gatherSend, METH_VARARGS, nullptr},
  {"_dist_gather_recv", (PyCFunction)THDPModule_gatherRecv, METH_VARARGS, nullptr},
  {"_dist_scatter_send", (PyCFunction)THDPModule_scatterSend, METH_VARARGS, nullptr},
  {"_dist_scatter_recv", (PyCFunction)THDPModule_scatterRecv, METH_VARARGS, nullptr},
  {"_dist_barrier", (PyCFunction)THDPModule_barrier, METH_O, nullptr},
  {"_dist_new_group", (PyCFunction)THDPModule_newGroup, METH_VARARGS, nullptr},
  {"_dist_request_is_completed", (PyCFunction)THDPModule_requestIsCompleted, METH_O, nullptr},
  {"_dist_request_wait", (PyCFunction)THDPModule_requestWait, METH_O, nullptr},
  {nullptr}
};

PyMethodDef* THDPModule_methods() {
  return _THDPModule_methods;
}
