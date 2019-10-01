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

C10_API TensorTypeSet tls_excluded_tensor_type_set();
C10_API TensorTypeSet tls_included_tensor_type_set();

// TODO: There is a semantic distinction here which is not made clear in
// the naming, and will need to be resolved at some point.  What
// is the difference between a mode like "Variable" and a mode
// like "Profiling" or "Tracing"?
//
//  - Variable is both a *tensor type* and a *mode*.  Variable is
//    selected if you have a "variable" tensor (which is most tensors
//    in PyTorch, except in weird cases like mobile), but after we handle
//    variable business, we turn it off (putting it in the excluded
//    set) so that we can process the insides without trigger variable.
//
//  - Profiling is a *mode* only.  There is no such thing as a
//    "profiling" tensor.  By default, profiling is turned off.  If
//    you turn it on, you want to turn it off once you instrument
//    a call (unless you're interested in getting nested profiling
//    ranges, in which case you delegate down the dispatch chain
//    for the immediate forwarding call, but don't turn it off.)
//
//    In the case of turning off profiling inside the body of
//    a profiled region, there is a degree of freedom: you can either
//    *disable* profiling, or you can *remove it from the enabled set*.
//    These two cases are distinguished by what happens if you "enable"
//    profiling inside of a disabled region (the reentrant case): in the former
//    case, you can reenable it; in the latter case, "enabling" inside a
//    disabled region wouldn't actually do anything.
//
//    {
//      EnableProfiling g1;
//      {
//        DisableProfiling g2;
//        {
//          EnableProfiling g3;
//          // is profiling on or off here?
//        }
//      }
//    }
//
//    There's probably a decent case to be made that "enabling" is a user
//    visible construct, while "disabling" is an implementation level construct.
//    So a user will have the tool (guards) to enable profiling, but not
//    disable it.
//
//    Unfortunately, I don't have a good answer for what should happen
//    in this case, especially for profiling.  For now, disabling profiling
//    simply removes it from the enabled set, because enabling profiling
//    MUST add it to the enabled set, and the disable function must
//    be an inverse (following the rule of minimal API.)
//
//  - Tracing is like profiling.  However, if you have turned off tracing
//    in the body of a function, it doesn't make very much sense to turn
//    it back on for the current tracing session.  This is because you've
//    traced a single node that represents all of its body, what are you
//    going to do with an inner trace?
//
//    (This is distinct from reentrant tracing, where you start a completely
//    different tracing session while in the midst of one tracing session.
//    This is not really something we support.)
//
// So for now, we just mix this all up.  But this all deserves a bigger
// rethink when it becomes relevant.

C10_API bool tls_variable_is_enabled();
C10_API void tls_variable_set_enabled(bool enabled);

C10_API bool TESTING_ONLY_tls_generic_mode_is_enabled();
C10_API void TESTING_ONLY_tls_generic_mode_set_enabled(bool enabled);

}} // namespace c10::impl
