#ifndef THP_WRAP_UTILS_INC
#define THP_WRAP_UTILS_INC

#define THPUtils_(NAME) TH_CONCAT_4(THP,Real,Utils_,NAME)

#define THPUtils_assert(cond, ...)                                             \
if (!(cond)) { THPUtils_setError(__VA_ARGS__); return NULL; }

bool THPUtils_checkLong(PyObject *index);
int THPUtils_getLong(PyObject *index, long *result);
long THPUtils_unpackLong(PyObject *index);
int THPUtils_getCallable(PyObject *arg, PyObject **result);
THLongStorage * THPUtils_getLongStorage(PyObject *args, int ignore_first=0);
void THPUtils_setError(const char *format, ...);
void THPUtils_invalidArguments(PyObject *given_args, const char *expected_args_desc);
PyObject * THPUtils_bytesFromString(const char *b);
bool THPUtils_checkBytes(PyObject *obj);
const char * THPUtils_bytesAsString(PyObject *bytes);

#define THStoragePtr TH_CONCAT_3(TH,Real,StoragePtr)
#define THTensorPtr  TH_CONCAT_3(TH,Real,TensorPtr)
#define THPStoragePtr TH_CONCAT_3(THP,Real,StoragePtr)
#define THPTensorPtr  TH_CONCAT_3(THP,Real,TensorPtr)

template<class T>
class THPPointer {
public:
  THPPointer(): ptr(nullptr) {};
  THPPointer(T *ptr): ptr(ptr) {};
  THPPointer(THPPointer &&p) { free(); ptr = p.ptr; p.ptr = nullptr; };

  ~THPPointer() { free(); };
  T * get() { return ptr; }
  T * release() { T *tmp = ptr; ptr = NULL; return tmp; }
  operator T*() { return ptr; }
  THPPointer& operator =(T *new_ptr) { free(); ptr = new_ptr; return *this; }
  T * operator ->() { return ptr; }
  operator bool() { return ptr != nullptr; }


private:
  void free();
  T *ptr = nullptr;
};

#include "generic/utils.h"
#include <TH/THGenerateAllTypes.h>

typedef THPPointer<PyObject> THPObjectPtr;
typedef THPPointer<THPGenerator> THPGeneratorPtr;

#endif

