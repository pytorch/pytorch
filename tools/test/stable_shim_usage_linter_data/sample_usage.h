// Sample file demonstrating correct and incorrect usage of versioned shim functions
// This file is used for testing the stable_shim_usage_linter

#include "sample_shim.h"

// Case 1: Correct usage - function with proper version guard
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
void correct_usage_v10() {
  simple_versioned_func(42); // OK: requires 2.10, have 2.10
}
#endif

// Case 2: ERROR - Unversioned call (no version guard at all)
void unversioned_call() {
  simple_versioned_func(42); // ERROR: requires 2.10, but no version guard
}

// Case 3: ERROR - Insufficient version (requires 2.10, but guarded with 2.9)
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
void insufficient_version() {
  simple_versioned_func(42); // ERROR: requires 2.10, but only have 2.9 guard
  old_function_1(123); // OK: requires 2.9, have 2.9
}
#endif

// Case 4: Correct usage - higher version than required is acceptable
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
void higher_version_ok() {
  simple_versioned_func(42); // OK: requires 2.10, have 2.11 (higher is fine)
  old_function_1(123); // OK: requires 2.9, have 2.11
}
#endif

// Case 5: Multiple errors in one block
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
void multiple_errors() {
  old_function_1(1); // OK: requires 2.9, have 2.9
  simple_versioned_func(2); // ERROR: requires 2.10, have 2.9
  callback_function_ptr cb; // ERROR: using versioned type requires 2.10, have 2.9
}
#endif

// Case 6: Nested version blocks - inner block is sufficient
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
void outer_block() {
  old_function_1(1); // OK: requires 2.9, have 2.9

#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  void inner_block() {
    simple_versioned_func(2); // OK: requires 2.10, have 2.10 in inner block
  }
#endif
}
#endif

// Case 7: Function call in #else branch (no version protection)
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
void if_branch() {
  simple_versioned_func(1); // OK
}
#else
void else_branch() {
  simple_versioned_func(2); // ERROR: #else branch has no version protection
}
#endif

// Case 8: Commented out calls should not trigger errors
void commented_calls() {
  // simple_versioned_func(42);  // This is commented, should not error
  /*
   * simple_versioned_func(42);  // This is in block comment
   */
}

// Case 9: #elif with version guard
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
void primary_path() {
  always_available_func(1); // OK: requires 2.11, have 2.11
}
#elif TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
void secondary_path() {
  simple_versioned_func(2); // OK: requires 2.10, have 2.10 in elif
  always_available_func(3); // ERROR: requires 2.11, but elif has 2.10
}
#endif

// Case 10: Unguarded calls at file scope
void more_unversioned() {
  old_function_1(1); // ERROR: requires 2.9, no guard
  old_function_2(nullptr); // ERROR: requires 2.9, no guard
}

// Case 11: Using type alias usage
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
void correct_handle_usage() {
  HandleType handle = nullptr; // OK: requires 2.10, have 2.10
}
#endif

// Case 12: ERROR - Using type alias with insufficient version
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_9_0
void insufficient_handle_usage() {
  HandleType handle = nullptr; // ERROR: requires 2.10, have 2.9
}
#endif

// Case 13: ERROR - Struct/class usage without version guard
void unversioned_struct_usage() {
  OpaqueHandle* ptr = nullptr; // ERROR: requires 2.10, no guard
  NewOpaqueStruct* new_ptr = nullptr; // ERROR: requires 2.11, no guard
}

// Case 14: Correct struct/class usage with proper version
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_11_0
void correct_new_struct_usage() {
  NewOpaqueStruct* ptr = nullptr; // OK: requires 2.11, have 2.11
  NewOpaqueClass* cls = nullptr; // OK: requires 2.11, have 2.11
  NewHandleType handle = nullptr; // OK: requires 2.11, have 2.11
}
#endif

// Case 15: ERROR - Struct usage with insufficient version
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
void insufficient_new_struct_usage() {
  NewOpaqueStruct* ptr = nullptr; // ERROR: requires 2.11, have 2.10
  NewOpaqueClass* cls = nullptr; // ERROR: requires 2.11, have 2.10
}
#endif
