#include <torch/csrc/distributed/c10d/store/TCPStoreTLS.hpp>

#include <torch/csrc/distributed/c10d/store/TCPStoreOpenSSL.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <string>
#include <string_view>
#include <vector>

#ifdef __linux__
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <c10/util/ScopeExit.h>
#include <c10/util/StringUtil.h>
#include <c10/util/error.h>
#include <fmt/format.h>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/logging.h>

namespace c10d::detail {
namespace {

// OpenSSL ABI constants, so no OpenSSL headers are needed at build time.
constexpr int kSslErrorWantRead = 2;
constexpr int kSslErrorZeroReturn = 6;
constexpr long kX509VOk = 0;

// Plaintext per SSL_write, bounding the BIO buffer, not the full response.
constexpr size_t kWriteChunk = 1 << 20;
// TLS records carry up to 16KiB of plaintext and less than 64 bytes overhead.
constexpr size_t kRecordSize = 16 * 1024;
constexpr size_t kRecordOverhead = 64;

std::string errorString() {
  std::string res;
  while (auto err = openssl().ERR_get_error()) {
    std::array<char, 256> buf{};
    openssl().ERR_error_string_n(err, buf.data(), buf.size());
    if (!res.empty()) {
      res += "; ";
    }
    res += buf.data();
  }
  return res;
}

int chunk(size_t length) {
  return static_cast<int>(std::min<size_t>(length, INT_MAX));
}

// Fails instead of prompting for the password of an encrypted key.
[[maybe_unused]] int noPassword(
    char* /*buf*/,
    int /*size*/,
    int /*rwflag*/,
    void* requested) {
  if (requested != nullptr) {
    *static_cast<bool*>(requested) = true;
  }
  return -1;
}

// Accept raw, NUL-free keys. OpenSSL 1.1.1 accepts at most 256 bytes
// (3.x allows 512). Use separate randomly generated keys for each protocol.
constexpr size_t kMinPskSize = 32;
constexpr size_t kMaxPskSize = 256;

#if defined(__linux__) && INTPTR_MAX == INT64_MAX
constexpr int kTls12Version = 0x0303;
constexpr int kTls13Version = 0x0304;
// Change the version suffix if the PSK handshake ever becomes incompatible.
constexpr std::string_view kPskIdentity = "torch.distributed.TCPStore.psk.v1";
constexpr std::array<unsigned char, 2> kTlsAes128GcmSha256 = {0x13, 0x01};

int rejectCertificate(int /*preverifyOk*/, void* /*storeCtx*/) {
  return 0;
}

// A new external PSK session for a handshake; OpenSSL takes ownership.
void* newPskSession(void* ssl) {
  const auto& lib = openssl();
  const auto* ctx = static_cast<const TLSContext*>(
      lib.SSL_CTX_get_ex_data(lib.SSL_get_SSL_CTX(ssl), 0));
  const auto& key = ctx->psk();
  const void* cipher = lib.SSL_CIPHER_find(ssl, kTlsAes128GcmSha256.data());
  void* session = lib.SSL_SESSION_new();
  if (cipher != nullptr && session != nullptr &&
      lib.SSL_SESSION_set1_master_key(
          session,
          reinterpret_cast<const unsigned char*>(key.data()),
          key.size()) == 1 &&
      lib.SSL_SESSION_set_cipher(session, cipher) == 1 &&
      lib.SSL_SESSION_set_protocol_version(session, kTls13Version) == 1) {
    return session;
  }
  lib.SSL_SESSION_free(session);
  return nullptr;
}

int usePskSession(
    void* ssl,
    const void* /*md*/,
    const unsigned char** id,
    size_t* idLength,
    void** session) {
  *session = newPskSession(ssl);
  if (*session == nullptr) {
    return 0;
  }
  *id = reinterpret_cast<const unsigned char*>(kPskIdentity.data());
  *idLength = kPskIdentity.size();
  return 1;
}

// An unknown identity leaves session null, failing the handshake.
int findPskSession(
    void* ssl,
    const unsigned char* id,
    size_t idLength,
    void** session) {
  *session = nullptr;
  if (std::string_view(reinterpret_cast<const char*>(id), idLength) !=
      kPskIdentity) {
    return 1;
  }
  *session = newPskSession(ssl);
  return *session != nullptr ? 1 : 0;
}
#endif

} // namespace

namespace {
std::string readPskFile(const std::string& path) {
#ifndef __linux__
  C10_THROW_ERROR(DistStoreError, "TCPStore TLS requires 64-bit Linux");
#else
  int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  TORCH_CHECK_WITH(
      DistStoreError, fd >= 0, "TCPStore TLS: cannot open PSK file");
  auto close = c10::make_scope_exit([&] { ::close(fd); });
  struct stat st{};
  TORCH_CHECK_WITH(
      DistStoreError,
      ::fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
          st.st_uid == ::geteuid() && (st.st_mode & 0077) == 0,
      "TCPStore TLS: PSK file must be a regular file owned by the current user with no group or other permissions");
  std::array<char, kMaxPskSize + 1> data{};
  auto wipe = c10::make_scope_exit([&] {
    volatile char* bytes = data.data();
    for (size_t i = 0; i < data.size(); ++i) {
      bytes[i] = 0;
    }
  });
  size_t size = 0;
  while (size < data.size()) {
    auto n = ::read(fd, data.data() + size, data.size() - size);
    if (n < 0 && errno == EINTR) {
      continue;
    }
    TORCH_CHECK_WITH(
        DistStoreError, n >= 0, "TCPStore TLS: cannot read PSK file");
    if (n == 0) {
      break;
    }
    size += n;
  }
  TORCH_CHECK_WITH(
      DistStoreError,
      size >= kMinPskSize && size <= kMaxPskSize,
      "TCPStore TLS: PSK must contain 32 to 256 bytes");
  return std::string(data.data(), size);
#endif
}

bool tlsRequired() {
  auto value = c10::utils::get_env("TORCH_TCPSTORE_TLS_REQUIRE");
  TORCH_CHECK_WITH(
      DistStoreError,
      !value || *value == "0" || *value == "1",
      "TORCH_TCPSTORE_TLS_REQUIRE must be 0 or 1");
  return value == "1";
}
} // namespace

TLSConfig tlsConfig(const TCPStoreOptions& opts) {
  const char* empty = nullptr;
  auto value = [&](const std::optional<std::string>& path, const char* name) {
    if (path && path->empty()) {
      empty = name;
    }
    TORCH_CHECK_WITH(
        DistStoreError,
        !path || path->find('\0') == std::string::npos,
        "TCPStore TLS: ",
        name,
        " must not contain NUL bytes");
    return path.value_or("");
  };
  const bool requireTls = tlsRequired();
  TLSConfig config;
  std::string pskFile;
  if (opts.tlsCert || opts.tlsKey || opts.tlsCa || opts.tlsPsk ||
      opts.tlsPskFile) {
    config = {
        value(opts.tlsCert, "tls_cert"),
        value(opts.tlsKey, "tls_key"),
        value(opts.tlsCa, "tls_ca"),
        value(opts.tlsPsk, "tls_psk")};
    pskFile = value(opts.tlsPskFile, "tls_psk_file");
  } else {
    auto env = [&](const char* name) {
      return value(c10::utils::get_env(name), name);
    };
    config = {
        env("TORCH_TCPSTORE_TLS_CERT"),
        env("TORCH_TCPSTORE_TLS_KEY"),
        env("TORCH_TCPSTORE_TLS_CA"),
        env("TORCH_TCPSTORE_TLS_PSK"),
        /*fromEnv=*/true};
    pskFile = env("TORCH_TCPSTORE_TLS_PSK_FILE");
  }
  if (!config.enabled() && pskFile.empty()) {
    TORCH_CHECK_WITH(
        DistStoreError,
        !requireTls,
        "TCPStore TLS is required but not configured");
    return config;
  }
  TORCH_CHECK_WITH(
      DistStoreError,
      empty == nullptr,
      "TCPStore TLS: ",
      empty,
      " must not be empty when TLS is enabled");
  if (!pskFile.empty()) {
    TORCH_CHECK_WITH(
        DistStoreError,
        !config.enabled(),
        "TCPStore TLS: PSK file cannot be combined with other TLS settings");
    config.psk = readPskFile(pskFile);
    TORCH_CHECK_WITH(
        DistStoreError,
        config.psk.find('\0') == std::string::npos,
        "TCPStore TLS: PSK must not contain NUL bytes");
  }
  if (!config.psk.empty()) {
    TORCH_CHECK_WITH(
        DistStoreError,
        config.cert.empty() && config.key.empty() && config.ca.empty(),
        config.fromEnv
            ? "TCPStore TLS: TORCH_TCPSTORE_TLS_PSK cannot be combined with "
              "TORCH_TCPSTORE_TLS_CERT, TORCH_TCPSTORE_TLS_KEY or "
              "TORCH_TCPSTORE_TLS_CA"
            : "TCPStore TLS: tls_psk cannot be combined with tls_cert, "
              "tls_key or tls_ca");
    TORCH_CHECK_WITH(
        DistStoreError,
        config.psk.size() >= kMinPskSize && config.psk.size() <= kMaxPskSize,
        "TCPStore TLS: PSK must contain 32 to 256 bytes");
    return config;
  }
  std::vector<std::string> missing;
  auto require =
      [&](const std::string& value, const char* opt, const char* env) {
        if (value.empty()) {
          missing.emplace_back(config.fromEnv ? env : opt);
        }
      };
  require(config.cert, "tls_cert", "TORCH_TCPSTORE_TLS_CERT");
  require(config.key, "tls_key", "TORCH_TCPSTORE_TLS_KEY");
  require(config.ca, "tls_ca", "TORCH_TCPSTORE_TLS_CA");
  TORCH_CHECK_WITH(
      DistStoreError,
      missing.empty(),
      "TCPStore TLS requires a certificate, key and CA; missing ",
      c10::Join(", ", missing),
      config.fromEnv ? " (from environment variables)"
                     : " (from TCPStore options)");
  return config;
}

std::shared_ptr<TLSContext> TLSContext::fromConfig(
    const TLSConfig& config,
    bool isServer) {
  if (!config.enabled()) {
    return nullptr;
  }
#if !defined(__linux__) || INTPTR_MAX != INT64_MAX
  C10_THROW_ERROR(DistStoreError, "TCPStore TLS requires 64-bit Linux");
#else
  constexpr int kSslCtrlSetMinProtoVersion = 123;
  constexpr int kSslCtrlSetMaxProtoVersion = 124;
  constexpr int kSslFiletypePem = 1;
  constexpr int kSslVerifyPeer = 0x01;
  constexpr int kSslVerifyFailIfNoPeerCert = 0x02;
  constexpr uint64_t kSslOpNoRenegotiation = 1U << 30;
  constexpr uint64_t kSslOpAllowNoDheKex = 1U << 10;
  constexpr uint64_t kSslOpPreferNoDheKex = 1ULL << 35;
  const auto& cert = config.cert;
  const auto& key = config.key;
  const auto& ca = config.ca;
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  void* ctx = ssl.SSL_CTX_new(ssl.TLS_method());
  TORCH_CHECK_WITH(
      DistStoreError, ctx, "TCPStore TLS: SSL_CTX_new failed: ", errorString());
  auto res = std::make_shared<TLSContext>(ctx, isServer);

  auto check = [](bool ok, const char* what, const std::string& path) {
    TORCH_CHECK_WITH(
        DistStoreError,
        ok,
        "TCPStore TLS: failed to load ",
        what,
        " ",
        path,
        ": ",
        errorString());
  };
  if (isServer) {
    // Session tickets are unused and would make a client socket readable
    // without a response.
    ssl.SSL_CTX_set_num_tickets(ctx, 0);
  }
  ssl.SSL_CTX_set_options(ctx, kSslOpNoRenegotiation);
  if (!config.psk.empty()) {
    res->psk_ = config.psk;
    // External PSKs require TLS 1.3. The cipher suite must match the PSK
    // session's hash. The key exchange still uses (EC)DHE.
    check(
        ssl.SSL_CTX_ctrl(
            ctx, kSslCtrlSetMinProtoVersion, kTls13Version, nullptr) == 1 &&
            ssl.SSL_CTX_ctrl(
                ctx, kSslCtrlSetMaxProtoVersion, kTls13Version, nullptr) == 1,
        "TLS version",
        "1.3");
    check(
        ssl.SSL_CTX_set_ciphersuites(ctx, "TLS_AES_128_GCM_SHA256") == 1,
        "cipher suite",
        "TLS_AES_128_GCM_SHA256");
    // Keep (EC)DHE, and so forward secrecy, even if openssl.cnf allows
    // PSK-only key exchange.
    ssl.SSL_CTX_clear_options(ctx, kSslOpAllowNoDheKex | kSslOpPreferNoDheKex);
    TORCH_CHECK_WITH(
        DistStoreError,
        ssl.SSL_CTX_set_ex_data(ctx, 0, res.get()) == 1,
        "TCPStore TLS: failed to initialize PSK context: ",
        errorString());
    if (isServer) {
      // Without a certificate, only PSK handshakes can succeed.
      ssl.SSL_CTX_set_psk_find_session_callback(ctx, findPskSession);
    } else {
      ssl.SSL_CTX_set_psk_use_session_callback(ctx, usePskSession);
      // A server certificate means the server did not accept the PSK.
      ssl.SSL_CTX_set_verify(ctx, kSslVerifyPeer, rejectCertificate);
    }
    return res;
  }
  check(
      ssl.SSL_CTX_ctrl(
          ctx, kSslCtrlSetMinProtoVersion, kTls12Version, nullptr) == 1,
      "minimum TLS version",
      "1.2");
  check(
      ssl.SSL_CTX_set_cipher_list(ctx, "ECDHE+AESGCM:ECDHE+CHACHA20") == 1,
      "TLS 1.2 cipher list",
      "ECDHE AEAD");
  bool encryptedKey = false;
  ssl.SSL_CTX_set_default_passwd_cb(ctx, noPassword);
  ssl.SSL_CTX_set_default_passwd_cb_userdata(ctx, &encryptedKey);
  check(
      ssl.SSL_CTX_use_certificate_chain_file(ctx, cert.c_str()) == 1,
      "certificate",
      cert);
  bool keyLoaded =
      ssl.SSL_CTX_use_PrivateKey_file(ctx, key.c_str(), kSslFiletypePem) == 1;
  ssl.SSL_CTX_set_default_passwd_cb_userdata(ctx, nullptr);
  if (encryptedKey) {
    ssl.ERR_clear_error();
    C10_THROW_ERROR(
        DistStoreError,
        fmt::format(
            "TCPStore TLS: encrypted private key {} is not supported", key));
  }
  check(keyLoaded, "private key", key);
  check(ssl.SSL_CTX_check_private_key(ctx) == 1, "matching private key", key);
  check(
      ssl.SSL_CTX_load_verify_locations(ctx, ca.c_str(), nullptr) == 1,
      "CA",
      ca);
  ssl.SSL_CTX_set_verify(
      ctx, kSslVerifyPeer | kSslVerifyFailIfNoPeerCert, nullptr);
  return res;
#endif
}

TLSContext::~TLSContext() {
  openssl().SSL_CTX_free(ctx_);
}

TLSSession::TLSSession(std::shared_ptr<TLSContext> ctx) : ctx_{std::move(ctx)} {
  const auto& ssl = openssl();
  ssl_ = ssl.SSL_new(ctx_->get());
  TORCH_CHECK_WITH(
      DistStoreError, ssl_, "TCPStore TLS: SSL_new failed: ", errorString());
  rbio_ = ssl.BIO_new(ssl.BIO_s_mem());
  wbio_ = ssl.BIO_new(ssl.BIO_s_mem());
  // SSL_free frees the BIOs once they are attached.
  ssl.SSL_set_bio(ssl_, rbio_, wbio_);
  if (!rbio_ || !wbio_) {
    auto err = errorString();
    ssl.SSL_free(ssl_);
    C10_THROW_ERROR(DistStoreError, "TCPStore TLS: BIO_new failed: " + err);
  }
  if (ctx_->isServer()) {
    ssl.SSL_set_accept_state(ssl_);
  } else {
    ssl.SSL_set_connect_state(ssl_);
  }
}

TLSSession::~TLSSession() {
  openssl().SSL_free(ssl_);
}

void TLSSession::throwError(int ret, const char* op) {
  const auto& ssl = openssl();
  if (ssl.SSL_get_error(ssl_, ret) == kSslErrorZeroReturn) {
    C10_THROW_ERROR(DistNetworkError, "TLS connection closed by peer");
  }
  auto msg = fmt::format("TCPStore TLS {} failed: {}", op, errorString());
  auto verify = ssl.SSL_get_verify_result(ssl_);
  if (verify != kX509VOk && ctx_->psk().empty()) {
    msg += fmt::format(
        " (certificate verify: {})", ssl.X509_verify_cert_error_string(verify));
  }
  msg += ctx_->psk().empty()
      ? ". Check that TLS is enabled on both client and server, that each "
        "certificate is signed by a CA trusted by the peer, that its purpose "
        "(extended key usage) allows both serverAuth and clientAuth, and that "
        "both use the same TLS mode (certificate or PSK)."
      : ". Check that the client and server both use TLS, the same TLS mode "
        "(certificate or PSK) and the same PSK.";
  C10_THROW_ERROR(DistStoreError, msg);
}

void TLSSession::feed(const void* data, size_t length) {
  auto ret = openssl().BIO_write(rbio_, data, chunk(length));
  TORCH_CHECK_WITH(
      DistStoreError,
      ret == static_cast<int>(length),
      "TCPStore TLS: BIO_write failed");
}

bool TLSSession::handshakeComplete() const {
  return openssl().SSL_is_init_finished(ssl_) == 1;
}

bool TLSSession::handshake() {
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  int ret = ssl.SSL_do_handshake(ssl_);
  if (ret == 1) {
    return true;
  }
  if (ssl.SSL_get_error(ssl_, ret) == kSslErrorWantRead) {
    return false;
  }
  throwError(ret, "handshake");
}

std::optional<size_t> TLSSession::read(void* data, size_t length) {
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  int ret = ssl.SSL_read(ssl_, data, chunk(length));
  if (ret > 0) {
    return ret;
  }
  auto error = ssl.SSL_get_error(ssl_, ret);
  if (error == kSslErrorWantRead) {
    return 0;
  }
  if (error == kSslErrorZeroReturn) {
    return std::nullopt;
  }
  throwError(ret, "read");
}

void TLSSession::shutdown() {
  const auto& ssl = openssl();
  ssl.ERR_clear_error();
  int ret = ssl.SSL_shutdown(ssl_);
  if (ret < 0) {
    throwError(ret, "shutdown");
  }
}

void TLSSession::write(
    const void* data,
    size_t length,
    std::vector<uint8_t>& out) {
  const auto& ssl = openssl();
  auto bytes = static_cast<const uint8_t*>(data);
  out.reserve(
      out.size() + length + (length / kRecordSize + 1) * kRecordOverhead);
  while (length > 0) {
    ssl.ERR_clear_error();
    int ret = ssl.SSL_write(
        ssl_, bytes, static_cast<int>(std::min(length, kWriteChunk)));
    if (ret <= 0) {
      throwError(ret, "write");
    }
    bytes += ret;
    length -= ret;
    takeOutput(out);
  }
}

void TLSSession::takeOutput(std::vector<uint8_t>& out) {
  const auto& ssl = openssl();
  while (auto pending = ssl.BIO_ctrl_pending(wbio_)) {
    auto offset = out.size();
    out.resize(offset + pending);
    int n = ssl.BIO_read(wbio_, out.data() + offset, chunk(pending));
    TORCH_CHECK_WITH(DistStoreError, n > 0, "TCPStore TLS: BIO_read failed");
    out.resize(offset + n);
  }
}

void TLSSession::flush(int socket, bool moreData) {
  std::vector<uint8_t> out;
  takeOutput(out);
  tcputil::sendBytes(socket, out.data(), out.size(), moreData);
}

bool TLSSession::recvSome(int socket, std::optional<Deadline> deadline) {
  while (deadline) {
    auto remaining = std::chrono::ceil<std::chrono::milliseconds>(
        *deadline - std::chrono::steady_clock::now());
    if (remaining <= std::chrono::milliseconds::zero()) {
      return false;
    }
    ::pollfd pfd{};
    pfd.fd = static_cast<decltype(pfd.fd)>(socket);
    pfd.events = POLLIN;
    auto timeout =
        static_cast<int>(std::clamp<int64_t>(remaining.count(), 0, INT_MAX));
#ifdef _WIN32
    int res = ::WSAPoll(&pfd, 1, timeout);
#else
    int res = ::poll(&pfd, 1, timeout);
#endif
    // Retry interrupted polls here so each attempt recomputes the deadline.
    if (res < 0 && errno != EINTR) {
      C10_THROW_ERROR(DistNetworkError, c10::utils::str_error(errno));
    }
    if (res > 0) {
      break;
    }
    if (std::chrono::steady_clock::now() >= *deadline) {
      return false;
    }
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  std::array<char, 16 * 1024> buf;
  ssize_t n = 0;
  SYSCHECK_ERR_RETURN_NEG1(n = ::recv(socket, buf.data(), buf.size(), 0))
  if (n == 0) {
    C10_THROW_ERROR(
        DistNetworkError,
        "Failed to recv, got 0 bytes. Connection was likely closed. "
        "Did the remote server shutdown or crash?");
  }
  feed(buf.data(), n);
  return true;
}

void TLSSession::handshake(int socket, std::optional<Deadline> deadline) {
  while (true) {
    bool done = false;
    try {
      done = handshake();
    } catch (const c10::DistStoreError&) {
      // Send the alert so the peer fails promptly.
      try {
        flush(socket);
      } catch (const std::exception& e) {
        C10D_DEBUG("Failed to send TLS alert: {}", e.what());
      }
      throw;
    }
    flush(socket);
    if (done) {
      return;
    }
    // Wants more input: all buffered input has been consumed.
    bool received = false;
    try {
      received = recvSome(socket, deadline);
    } catch (const c10::DistNetworkError& e) {
      C10_THROW_ERROR(
          DistNetworkError,
          fmt::format(
              "Connection closed during TLS handshake. Is TLS enabled on the peer? {}",
              e.what_without_backtrace()));
    }
    if (!received) {
      C10_THROW_ERROR(
          DistNetworkError,
          "TLS handshake timed out. Is TLS enabled on the peer?");
    }
  }
}

bool TLSSession::waitForInput(int socket, Deadline deadline) {
  const auto& ssl = openssl();
  while (true) {
    uint8_t byte = 0;
    ssl.ERR_clear_error();
    int ret = ssl.SSL_peek(ssl_, &byte, 1);
    if (ret > 0) {
      return true;
    }
    if (ssl.SSL_get_error(ssl_, ret) != kSslErrorWantRead) {
      throwError(ret, "read");
    }
    flush(socket);
    if (!recvSome(socket, deadline)) {
      return false;
    }
  }
}

void TLSSession::sendBytes(
    int socket,
    const void* data,
    size_t length,
    bool moreData) {
  auto bytes = static_cast<const uint8_t*>(data);
  while (length > 0) {
    auto n = std::min(length, kWriteChunk);
    std::vector<uint8_t> out;
    write(bytes, n, out);
    bytes += n;
    length -= n;
    tcputil::sendBytes(socket, out.data(), out.size(), moreData || length > 0);
  }
}

void TLSSession::recvBytes(int socket, void* data, size_t length) {
  auto bytes = static_cast<uint8_t*>(data);
  while (length > 0) {
    auto n = read(bytes, length);
    TORCH_CHECK_WITH(
        DistNetworkError, n.has_value(), "TLS connection closed by peer");
    if (*n == 0) {
      flush(socket);
      recvSome(socket);
      continue;
    }
    bytes += *n;
    length -= *n;
  }
}

void SocketIO::sendBytes(const void* data, size_t length, bool moreData) const {
  if (tls == nullptr) {
    tcputil::sendBytes(
        socket, static_cast<const uint8_t*>(data), length, moreData);
    return;
  }
  tls->sendBytes(socket, data, length, moreData);
}

void SocketIO::recvBytes(void* data, size_t length) const {
  if (tls == nullptr) {
    tcputil::recvBytes(socket, static_cast<uint8_t*>(data), length);
    return;
  }
  tls->recvBytes(socket, data, length);
}

} // namespace c10d::detail
