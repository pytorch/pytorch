#pragma once
#include <c10/macros/Export.h>
#include <functional>

namespace c10 {
// Used for feature rollouts where we need to be
// able to stop the rollout quickly, or provide OSS
// users who are using nightlies a way to work around
// new features while we resolve what is broken.

// feature_name, by convention, should have the form
// TORCH_FEATURE_{FEATURE_NAME}_ENABLED.

// the_namespace - when a custom resolver is set,
// the_namespace allows features to be grouped by subcategory
// it is ignored by the default provider.

// default_value - Then value to return when nothing is set
// on platforms where it is acceptable to have a default.
// Some platforms (e.g. justknobs) do not have default
// values, and will need to be configured when a new rollout is 
// added so it can be tracked.

// For feature rollouts, the default should be true
// to gather test signal, and should only be disabled
// via the feature system or environment variable.

// WARNING: Do NOT call this function at module import time.
// JK is used internally to implement feature enabled. JK is not
// fork safe and you will break anyone who forks the process and then
// hits JK again.

// ADDING A NEW FEATURE: in addition to adding the check into the code,
// a new feature must be created in justknobs with the same name before being
// imported into fbcode. Otherwise the check will fail internally.
// There are no default values by design.

// Default: true
bool C10_EXPORT justknobs_check(const char* the_namespace, const char* feature_name, bool default_value);
bool C10_EXPORT justknobs_check_default(const char* the_namespace, const char* feature_name, bool default_value);

// Deployments may want to change how they control features.
// They should call SetFeatureResolver, to change the implementation
// of FeatureEnabled from the default.
void C10_EXPORT set_justknobs_check_resolver(std::function<bool(const char*, const char*, bool)> resolver);

} // namespace c10
