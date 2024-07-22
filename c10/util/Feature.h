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

// By default, FeatureEnabled will look for the environment variable
// with this name to decide whether the feature is on.
// Since the purpose is to disable features when they
// turn out not to work in certain cases, when the string is
// not found, we default to true.

// WARNING: Do NOT call this function at module import time.
// JK is used internally to implement feature enabled. JK is not
// fork safe and you will break anyone who forks the process and then
// hits JK again.

// Default: true
bool C10_EXPORT FeatureEnabled(const char* the_namespace, const char* feature_name);
bool C10_EXPORT FeatureEnabledDefaultResolver(const char* the_namespace, const char* feature_name);

// Deployments may want to change how they control features.
// They should call SetFeatureResolver, to change the implementation
// of FeatureEnabled from the default.
void C10_EXPORT SetFeatureResolver(std::function<bool(const char*, const char*)> resolver);

} // namespace c10
