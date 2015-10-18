#ifndef CAFFE2_CORE_TYPEID_H_
#define CAFFE2_CORE_TYPEID_H_

#include <map>
#include <typeinfo>

#include "caffe2/core/common.h"

namespace caffe2 {

typedef intptr_t CaffeTypeId;

/**
 * TypeMeta is a thin class that allows us to store the type of a container such
 * as a blob, or the data type of a tensor, with a unique run-time id. It also
 * stores some additional data such as the item size and the name of the type
 * for run-time inspection.
 */
class TypeMeta {
 public:
  /** Create a dummy TypeMeta object. To create a TypeMeta object for a specific
   * type, use TypeMeta::Make<T>().
   */
  TypeMeta() : id_(0), itemsize_(0), name_("Unknown type") {}
  /**
   * Copy constructor.
   */
  TypeMeta(const TypeMeta& src)
      : id_(src.id_), itemsize_(src.itemsize_), name_(src.name_) {}
  /**
   * Assignment operator.
   */
  TypeMeta& operator=(const TypeMeta& src) {
    if (this == &src) return *this;
    id_ = src.id_;
    itemsize_ = src.itemsize_;
    name_ = src.name_;
    return *this;
  }

  /**
   * Returns the type id.
   */
  inline const CaffeTypeId& id() const { return id_; }
  /**
   * Returns the size of the item.
   */
  inline const size_t& itemsize() const { return itemsize_; }
  /**
   * Returns a printable name for the type.
   */
  inline const char* const& name() const { return name_; }
  bool operator==(const TypeMeta& other) const { return (id_ == other.id_); }
  bool operator!=(const TypeMeta& other) const { return (id_ != other.id_); }

  template <typename T>
  inline bool Match() const { return (id_ == Id<T>()); }

  // Below are static functions that can be called by passing a specific type.

  /**
   * Returns the unique id for the given type T. The id is unique for the type T
   * in the sense that for any two different types, their id are different; for
   * the same type T, the id remains the same over different calls of the
   * function. However, this is not guaranteed over different runs, as the id
   * is generated during run-time. Do NOT serialize the id for storage.
   */
  template <typename T>
  static CaffeTypeId Id() {
    static bool type_id_bit[1];
    return reinterpret_cast<CaffeTypeId>(type_id_bit);
  }
  /**
   * Returns the item size of the type. This is equivalent to sizeof(T).
   */
  template <typename T>
  static size_t ItemSize() { return sizeof(T); }
  /**
   * Returns the printable name of the type.
   */
  template <typename T>
  static const char* Name() { return typeid(T).name(); }
  /**
   * Returns a TypeMeta object that corresponds to the typename T.
   */
  template <typename T>
  static TypeMeta Make() {
    return TypeMeta(Id<T>(), ItemSize<T>(), Name<T>());
  }

 private:
  // TypeMeta can only be created by Make, making sure that we do not
  // create incorrectly mixed up TypeMeta objects.
  TypeMeta(CaffeTypeId i, size_t s, const char* n)
      : id_(i), itemsize_(s), name_(n) {}
  CaffeTypeId id_;
  size_t itemsize_;
  const char* name_;
};

}  // namespace caffe2

#endif  // CAFFE2_CORE_TYPEID_H_
