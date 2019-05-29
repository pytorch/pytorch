#pragma once

#include <c10/macros/Macros.h>
#include <ATen/core/dispatch/Dispatcher.h>

namespace c10 {
namespace detail {

template<class Data> class OperatorMetadataType;

template<class Data>
class OpRegistrationListenerForMetadata final : public OpRegistrationListener {
public:
  explicit OpRegistrationListenerForMetadata(OperatorMetadataType<Data>* metadataType)
  : metadataType_(metadataType) {}

  void onOperatorRegistered(const OperatorHandle& op) override {
    // no-op
  }

  void onOperatorDeregistered(const OperatorHandle& op) override {
    metadataType_->removeOperator(op);
  }

private:
  OperatorMetadataType<Data>* metadataType_;
};

/**
 * OperatorMetadataType is an extensible way to store metadata for an operator
 * without the c10 dispatcher needing to know about the metadata.
 * New kinds of metadata can easily be added externally.
 */

template<class Data>
class CAFFE2_API OperatorMetadataType final {
public:
  explicit OperatorMetadataType()
  : metadataForOperators_()
  , listenerRegistrationHandle_(
      Dispatcher::singleton().addRegistrationListener(
        guts::make_unique<detail::OpRegistrationListenerForMetadata<Data>>(this))
    ) {}

  void set(OperatorHandle op, Data&& metadata) {
    metadataForOperators_.insert_or_assign(op, std::move(metadata));
  }

  c10::optional<const Data*> get(OperatorHandle op) {
    auto found = metadataForOperators_.find(op);
    if (found == metadataForOperators_.end()) {
      return c10::nullopt;
    }
    return &found->second;
  }

  void removeOperator(OperatorHandle op) {
    metadataForOperators_.erase(op);
  }

private:
  ska::flat_hash_map<OperatorHandle, Data> metadataForOperators_;
  RegistrationHandleRAII listenerRegistrationHandle_;
};

template<class Data> CAFFE2_API OperatorMetadataType<Data>& operatorMetadataTypeSingleton();

}

/**
 * Call this macro to register a new metadata type for operators.
 * After calling this on a type T, you can use get_op_metadata<T>(op)
 * and set_op_metadata<T>(op, data) to load and store metadata.
 *
 * Please use your own custom types (e.g. structs) for storing metadata,
 * instead of commonly used types like std::string or int, otherwise you're
 * likely to clash with other code registering metadata for std::string.
 *
 * This macro must be called from the top level namespace, i.e. outside
 * of any namespaces.
 */
#define TORCH_DEFINE_OPERATOR_METADATA_TYPE(Type)                             \
  namespace c10 { namespace detail {                                          \
    template<>                                                                \
    C10_EXPORT OperatorMetadataType<Type>&                                    \
        operatorMetadataTypeSingleton<Type>() {                               \
      static OperatorMetadataType<Type> singleton;                            \
      return singleton;                                                       \
    }                                                                         \
  }}

/**
 * Get the metadata of type Data from the operator op.
 * If metadata of this type was registered for this operator before
 * using set_op_metadata<Data>(op, data), then this will return that metadata.
 * Otherwise, returns c10::nullopt.
 */
template<class Data>
inline c10::optional<const Data*> get_op_metadata(OperatorHandle op) {
  return detail::operatorMetadataTypeSingleton<Data>().get(op);
}

/**
 * Set the metadata of type Data from the operator op.
 * After setting the metadata for an operator using
 * set_op_metadata<Data>(op, data), it can be retrieved using
 * get_op_metadata<Data>(op).
 */
template<class Data>
inline void set_op_metadata(OperatorHandle op, Data data) {
  detail::operatorMetadataTypeSingleton<Data>().set(op, std::move(data));
}

}
