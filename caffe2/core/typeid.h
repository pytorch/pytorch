#ifndef CAFFE2_CORE_TYPEID_H_
#define CAFFE2_CORE_TYPEID_H_

#include <map>
#include <typeinfo>

#include "caffe2/core/common.h"
#include "glog/logging.h"

namespace caffe2 {
namespace internal {

static_assert(sizeof(void*) <= sizeof(int64_t),
              "This does not happen often, but int64_t is not enough for "
              "pointers on this platform.");
typedef int64_t TypeId;
extern std::map<TypeId, string> g_caffe2_type_name_map;
const TypeId gUnknownType = 0;

template <class T>
class TypeIdRegisterer {
 public:
  TypeIdRegisterer() {
    CHECK_EQ(g_caffe2_type_name_map.count(id()), 0)
        << "Registerer instantiated twice.";
    g_caffe2_type_name_map[id()] = typeid(T).name();
  }
  inline TypeId id() {
    return reinterpret_cast<TypeId>(type_id_bit);
  }

 private:
  bool type_id_bit[1];
};

// id = TypeId<T>() gives a unique type id for the given class, which can be
// verified by IsType<T>(id). This allows us to check the type of object
// pointers during run-time.
template <class T>
TypeId GetTypeId() {
  static TypeIdRegisterer<T> reg;
  return reg.id();
}

template <class T>
inline bool IsTypeId(TypeId id) {
  return (id == GetTypeId<T>());
}

inline string TypeName(TypeId id) {
  if (id == gUnknownType) return "UNKNOWN";
  return g_caffe2_type_name_map[id];
}

template <class T>
inline string TypeName() {
  return TypeName(GetTypeId<T>());
}

}  // namespace internal
}  // namespace caffe2

#endif  // CAFFE2_CORE_TYPEID_H_
