#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <ATen/DynamicLibrary.h>
#include <c10/util/Exception.h>
#include <fmt/format.h>

namespace c10d::detail {

// Runtime OpenSSL bindings; no OpenSSL headers or link dependency required.
using PasswordCallback = int (*)(char*, int, int, void*);
using VerifyCallback = int (*)(int, void*);
using PskUseSessionCallback =
    int (*)(void*, const void*, const unsigned char**, size_t*, void**);
using PskFindSessionCallback =
    int (*)(void*, const unsigned char*, size_t, void**);

#define TCPSTORE_OPENSSL_FUNCTIONS(_)                                        \
  _(unsigned long, OpenSSL_version_num, ())                                  \
  _(const char*, OpenSSL_version, (int))                                     \
  _(const void*, TLS_method, ())                                             \
  _(void*, SSL_CTX_new, (const void*))                                       \
  _(void, SSL_CTX_free, (void*))                                             \
  _(long, SSL_CTX_ctrl, (void*, int, long, void*))                           \
  _(int, SSL_CTX_use_certificate_chain_file, (void*, const char*))           \
  _(int, SSL_CTX_use_PrivateKey_file, (void*, const char*, int))             \
  _(int, SSL_CTX_check_private_key, (const void*))                           \
  _(void, SSL_CTX_set_default_passwd_cb, (void*, PasswordCallback))          \
  _(void, SSL_CTX_set_default_passwd_cb_userdata, (void*, void*))            \
  _(int, SSL_CTX_load_verify_locations, (void*, const char*, const char*))   \
  _(void, SSL_CTX_set_verify, (void*, int, VerifyCallback))                  \
  _(int, SSL_CTX_set_num_tickets, (void*, size_t))                           \
  _(uint64_t, SSL_CTX_set_options, (void*, uint64_t))                        \
  _(uint64_t, SSL_CTX_clear_options, (void*, uint64_t))                      \
  _(int, SSL_CTX_set_cipher_list, (void*, const char*))                      \
  _(int, SSL_CTX_set_ciphersuites, (void*, const char*))                     \
  _(void,                                                                    \
    SSL_CTX_set_psk_use_session_callback,                                    \
    (void*, PskUseSessionCallback))                                          \
  _(void,                                                                    \
    SSL_CTX_set_psk_find_session_callback,                                   \
    (void*, PskFindSessionCallback))                                         \
  _(int, SSL_CTX_set_ex_data, (void*, int, void*))                           \
  _(void*, SSL_CTX_get_ex_data, (const void*, int))                          \
  _(void*, SSL_get_SSL_CTX, (const void*))                                   \
  _(const void*, SSL_CIPHER_find, (void*, const unsigned char*))             \
  _(void*, SSL_SESSION_new, ())                                              \
  _(void, SSL_SESSION_free, (void*))                                         \
  _(int, SSL_SESSION_set1_master_key, (void*, const unsigned char*, size_t)) \
  _(int, SSL_SESSION_set_cipher, (void*, const void*))                       \
  _(int, SSL_SESSION_set_protocol_version, (void*, int))                     \
  _(void*, SSL_new, (void*))                                                 \
  _(void, SSL_free, (void*))                                                 \
  _(void, SSL_set_bio, (void*, void*, void*))                                \
  _(void, SSL_set_connect_state, (void*))                                    \
  _(void, SSL_set_accept_state, (void*))                                     \
  _(int, SSL_shutdown, (void*))                                              \
  _(int, SSL_is_init_finished, (const void*))                                \
  _(int, SSL_do_handshake, (void*))                                          \
  _(int, SSL_read, (void*, void*, int))                                      \
  _(int, SSL_peek, (void*, void*, int))                                      \
  _(int, SSL_write, (void*, const void*, int))                               \
  _(int, SSL_get_error, (const void*, int))                                  \
  _(long, SSL_get_verify_result, (const void*))                              \
  _(const void*, BIO_s_mem, ())                                              \
  _(void*, BIO_new, (const void*))                                           \
  _(int, BIO_read, (void*, void*, int))                                      \
  _(int, BIO_write, (void*, const void*, int))                               \
  _(size_t, BIO_ctrl_pending, (void*))                                       \
  _(unsigned long, ERR_get_error, ())                                        \
  _(void, ERR_error_string_n, (unsigned long, char*, size_t))                \
  _(void, ERR_clear_error, ())                                               \
  _(const char*, X509_verify_cert_error_string, (long))

struct OpenSSL {
#define TCPSTORE_DECLARE(ret, name, args) ret(*name) args = nullptr;
  TCPSTORE_OPENSSL_FUNCTIONS(TCPSTORE_DECLARE)
#undef TCPSTORE_DECLARE

  OpenSSL() {
    std::string errors;
    for (const char* name : {"libssl.so.3", "libssl.so.1.1"}) {
      try {
        // libcrypto symbols resolve through this libssl's own dependency, so
        // the versions always match.
        at::DynamicLibrary lib(name);
#define TCPSTORE_LOAD(ret, name, args) \
  name = reinterpret_cast<ret(*) args>(lib.sym(#name));
        TCPSTORE_OPENSSL_FUNCTIONS(TCPSTORE_LOAD)
#undef TCPSTORE_LOAD
        auto version = OpenSSL_version(0);
        if (OpenSSL_version_num() < 0x10101000L || version == nullptr ||
            std::string_view(version).find("OpenSSL ") != 0) {
          errors += "; unsupported OpenSSL version or vendor";
          continue;
        }
        // Only a fully validated candidate gets a process-lifetime reference.
        at::DynamicLibrary keepAlive(name, nullptr, /*leak_handle=*/true);
        return;
      } catch (const c10::DynamicLibraryError& e) {
        errors += fmt::format("; {}", e.what_without_backtrace());
      }
    }
    C10_THROW_ERROR(
        DistStoreError,
        "TCPStore TLS requires OpenSSL >= 1.1.1 (libssl.so.3 or libssl.so.1.1)" +
            errors);
  }
};

inline const OpenSSL& openssl() {
  static const OpenSSL lib;
  return lib;
}

#undef TCPSTORE_OPENSSL_FUNCTIONS

} // namespace c10d::detail
