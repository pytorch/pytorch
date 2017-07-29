#include "DynamicTypes.h"

#include "THP.h"
#include <vector>
#include <unordered_map>

#ifdef WITH_CUDA
#include <THC/THC.h>
#include <THCS/THCS.h>
#endif

namespace torch {

static std::unordered_map<std::string, at::ScalarType> attype_names = {
  {"Float", at::kFloat},
  {"Double", at::kDouble},
  {"Half", at::kHalf},
  {"Byte", at::kByte},
  {"Char", at::kChar},
  {"Short", at::kShort},
  {"Int", at::kInt},
  {"Long", at::kLong},
};

static std::unordered_map<PyTypeObject*, at::Type*> pytype_to_attype;
static std::unordered_map<at::Type*, PyTypeObject*> attype_to_pytype;

void registerPyTypeObject(PyTypeObject *pytype, const std::string& name, bool is_cuda, bool is_sparse)
{
  at::Backend device;
  if(is_cuda) {
    if(is_sparse){
      device = at::kSparseCUDA;
    } else {
      device = at::kCUDA;
    }
  } else {
    if(is_sparse){
      device = at::kSparseCPU;
    } else {
      device = at::kCPU;
    }
  }

  if(!(is_sparse && name == "Half")) {
    at::Type * attype = &at::getType(device,attype_names.at(name));
    pytype_to_attype[pytype] = attype;
    attype_to_pytype[attype] = pytype;
  }
}

PyTypeObject* getPyTypeObject(const at::Tensor& tensor)
{
  if(!tensor.defined())
    throw std::invalid_argument("trying to get type of undefined at::Tensor");
  if(attype_to_pytype.count(&tensor.type()) == 0)
    throw std::invalid_argument("unsupported Tensor type.");
  return attype_to_pytype.at(&tensor.type());
}

at::Tensor createTensor(PyObject *data)
{
  auto tensor_type = pytype_to_attype.at(Py_TYPE(data));
  auto tensor = ((THPVoidTensor *)data)->cdata;
  return tensor_type->unsafeTensorFromTH(tensor, true); // Calls retain on underlying TH Tensor
}
PyObject* createPyObject(at::Tensor& tensor)
{
  auto type = getPyTypeObject(tensor);
  PyObject *obj = type->tp_alloc(type, 0);
  if (obj) {
    // Retain underlying TH Tensor
    ((THPVoidTensor*)obj)->cdata = (THVoidTensor *)tensor.unsafeGetTH(true);
  }
  return obj;
}

}  // namespace
