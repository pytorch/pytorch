// override functions from here: https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html

// generated with the script:
// for i in $(with-proxy curl https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html | grep -oP '\<strong\>\K__atomic[^<]*'); do echo "#define $i(...) (_nano_tracking_log(\"$i\"), $i(__VA_ARGS__))"; done

extern "C" {
void __attribute__((weak)) nano_tracking_log(const char* event);
}

static inline void _nano_tracking_log(const char* event) {
  if (&nano_tracking_log) {
    nano_tracking_log(event);
  }
}

#define __atomic_load_n(...) (_nano_tracking_log("__atomic_load_n"), __atomic_load_n(__VA_ARGS__))
#define __atomic_load(...) (_nano_tracking_log("__atomic_load"), __atomic_load(__VA_ARGS__))
#define __atomic_store_n(...) (_nano_tracking_log("__atomic_store_n"), __atomic_store_n(__VA_ARGS__))
#define __atomic_store(...) (_nano_tracking_log("__atomic_store"), __atomic_store(__VA_ARGS__))
#define __atomic_exchange_n(...) (_nano_tracking_log("__atomic_exchange_n"), __atomic_exchange_n(__VA_ARGS__))
#define __atomic_exchange(...) (_nano_tracking_log("__atomic_exchange"), __atomic_exchange(__VA_ARGS__))
#define __atomic_compare_exchange_n(...) (_nano_tracking_log("__atomic_compare_exchange_n"), __atomic_compare_exchange_n(__VA_ARGS__))
#define __atomic_compare_exchange(...) (_nano_tracking_log("__atomic_compare_exchange"), __atomic_compare_exchange(__VA_ARGS__))
#define __atomic_add_fetch(...) (_nano_tracking_log("__atomic_add_fetch"), __atomic_add_fetch(__VA_ARGS__))
#define __atomic_sub_fetch(...) (_nano_tracking_log("__atomic_sub_fetch"), __atomic_sub_fetch(__VA_ARGS__))
#define __atomic_and_fetch(...) (_nano_tracking_log("__atomic_and_fetch"), __atomic_and_fetch(__VA_ARGS__))
#define __atomic_xor_fetch(...) (_nano_tracking_log("__atomic_xor_fetch"), __atomic_xor_fetch(__VA_ARGS__))
#define __atomic_or_fetch(...) (_nano_tracking_log("__atomic_or_fetch"), __atomic_or_fetch(__VA_ARGS__))
#define __atomic_nand_fetch(...) (_nano_tracking_log("__atomic_nand_fetch"), __atomic_nand_fetch(__VA_ARGS__))
#define __atomic_fetch_add(...) (_nano_tracking_log("__atomic_fetch_add"), __atomic_fetch_add(__VA_ARGS__))
#define __atomic_fetch_sub(...) (_nano_tracking_log("__atomic_fetch_sub"), __atomic_fetch_sub(__VA_ARGS__))
#define __atomic_fetch_and(...) (_nano_tracking_log("__atomic_fetch_and"), __atomic_fetch_and(__VA_ARGS__))
#define __atomic_fetch_xor(...) (_nano_tracking_log("__atomic_fetch_xor"), __atomic_fetch_xor(__VA_ARGS__))
#define __atomic_fetch_or(...) (_nano_tracking_log("__atomic_fetch_or"), __atomic_fetch_or(__VA_ARGS__))
#define __atomic_fetch_nand(...) (_nano_tracking_log("__atomic_fetch_nand"), __atomic_fetch_nand(__VA_ARGS__))
#define __atomic_test_and_set(...) (_nano_tracking_log("__atomic_test_and_set"), __atomic_test_and_set(__VA_ARGS__))
#define __atomic_clear(...) (_nano_tracking_log("__atomic_clear"), __atomic_clear(__VA_ARGS__))
#define __atomic_thread_fence(...) (_nano_tracking_log("__atomic_thread_fence"), __atomic_thread_fence(__VA_ARGS__))
#define __atomic_signal_fence(...) (_nano_tracking_log("__atomic_signal_fence"), __atomic_signal_fence(__VA_ARGS__))
#define __atomic_always_lock_free(...) (_nano_tracking_log("__atomic_always_lock_free"), __atomic_always_lock_free(__VA_ARGS__))
#define __atomic_is_lock_free(...) (_nano_tracking_log("__atomic_is_lock_free"), __atomic_is_lock_free(__VA_ARGS__))
