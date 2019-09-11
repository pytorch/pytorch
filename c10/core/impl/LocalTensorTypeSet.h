#include <c10/core/TensorTypeSet.h>

// TLS management for TensorTypeSet
//
// This manages thread-local TensorTypeSet of excluded keys which disqualify
// tensor types from dispatch.  Keys which are in this set, even if they appear
// in a list of potential valid keys on a tensor, are not considered for
// dispatch.  This is used to, for example, turn off autograd after we have
// handled autograd for a top-level element.
//
// Originally, I implemented this as storing the inverted set, but
// TLS is defined to be zero-initialized, so this doesn't actually work
// (you want the set to be -1 initialized).

namespace c10 {
namespace impl {

C10_API bool tls_variable_is_enabled();
C10_API void tls_variable_set_enabled(bool enabled);
C10_API TensorTypeSet tls_excluded_tensor_type_set();

}} // namespace c10::impl
