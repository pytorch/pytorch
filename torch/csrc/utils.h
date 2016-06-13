#ifndef THP_WRAP_UTILS_INC
#define THP_WRAP_UTILS_INC

#define THPUtils_(NAME) TH_CONCAT_4(THP,Real,Utils_,NAME)

int THPUtils_getLong(PyObject *index, long *result);
int THPUtils_getCallable(PyObject *arg, PyObject **result);
THLongStorage * THPUtils_getLongStorage(PyObject *args, int ignore_first=0);
void THPUtils_setError(const char *format, ...);

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
  T *ptr;
};

#include "generic/utils.h"
#include <TH/THGenerateAllTypes.h>

#define THPUtils_assert(cond, ...)                                             \
if (!(cond)) { THPUtils_setError(__VA_ARGS__); return NULL; }

#endif

