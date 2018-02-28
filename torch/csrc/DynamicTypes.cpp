#include <Python.h>

#include "DynamicTypes.h"
#include "PythonTypes.h"
#include "THP.h"
#include "Exceptions.h"

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

static std::unordered_map<at::Type*, PyTypeObject*> attype_to_py_storage_type;
static std::unordered_map<PyTypeObject*, at::Type*> py_storage_type_to_attype;
static std::unordered_map<const at::Type*, THPDtype*> attype_to_dtype;

static at::Backend get_backend(bool is_cuda, bool is_sparse) {
  if (is_cuda) {
    if (is_sparse){
      return at::kSparseCUDA;
    } else {
      return at::kCUDA;
    }
  } else {
    if (is_sparse){
      return at::kSparseCPU;
    } else {
      return at::kCPU;
    }
  }
}

static at::Type* get_type(const std::string& name, bool is_cuda, bool is_sparse) {
  if (is_sparse && name == "Half") {
    return nullptr;
  }
  at::Backend backend = get_backend(is_cuda, is_sparse);
  return &at::getType(backend, attype_names.at(name));
}

void registerStoragePyTypeObject(PyTypeObject *pytype, const std::string& name, bool is_cuda, bool is_sparse)
{
  auto attype = get_type(name, is_cuda, is_sparse);
  if (attype) {
    attype_to_py_storage_type[attype] = pytype;
    py_storage_type_to_attype[pytype] = attype;
  }
}

void registerDtypeObject(THPDtype *dtype, at::Type& type) {
  attype_to_dtype[&type] = dtype;
}

static PyTypeObject* getPyTypeObject(const at::Storage& storage)
{
  auto it = attype_to_py_storage_type.find(&storage.type());
  if (it != attype_to_py_storage_type.end()) {
    return it->second;
  }
  throw std::invalid_argument("unsupported Storage type");
}

THPDtype* getDtype(const at::Type& type) {
  auto it = attype_to_dtype.find(&type);
  if (it != attype_to_dtype.end()) {
    return it->second;
  }
  throw std::invalid_argument("unsupported at::Type");
}

PyObject* createPyObject(const at::Storage& storage)
{
  auto type = getPyTypeObject(storage);
  auto obj = THPObjectPtr(type->tp_alloc(type, 0));
  if (!obj) throw python_error();
  ((THPVoidStorage*)obj.get())->cdata = (THVoidStorage *)storage.unsafeGetTH(true);
  return obj.release();
}

bool isStorage(PyObject* obj)
{
  auto it = py_storage_type_to_attype.find(Py_TYPE(obj));
  return it != py_storage_type_to_attype.end();
}
std::unique_ptr<at::Storage> createStorage(PyObject* obj)
{
  auto it = py_storage_type_to_attype.find(Py_TYPE(obj));
  if (it == py_storage_type_to_attype.end()) {
    throw TypeError("not a storage '%s'", Py_TYPE(obj)->tp_name);
  }
  auto& type = *it->second;
  return type.unsafeStorageFromTH(((THPVoidStorage*)obj)->cdata, true);
}

}  // namespace
