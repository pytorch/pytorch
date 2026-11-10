#include <torch/csrc/distributed/c10d/store/TCPStoreAuth.hpp>

#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <torch/csrc/distributed/c10d/exception.h>

#include <filesystem>
#include <system_error>
#include <tuple>

#ifdef __linux__
#include <ATen/DynamicLibrary.h>
#include <c10/util/error.h>
#include <poll.h>
#include <sys/socket.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstring>
#include <string>
#include <string_view>
#endif

namespace c10d::detail {

#ifndef __linux__

struct TCPStoreAuth::Impl {};

void TCPStoreAuth::clientHandshake(int, std::chrono::steady_clock::time_point)
    const {}
std::vector<uint8_t> TCPStoreAuth::serverHello(
    const std::vector<uint8_t>&,
    Nonce&) const {
  return {};
}
void TCPStoreAuth::verifyClient(
    const std::vector<uint8_t>&,
    const Nonce&,
    const std::vector<uint8_t>&) const {}

#else

namespace {

// Opaque OpenSSL types. libcrypto is loaded at runtime so that no OpenSSL
// headers or link-time dependency are needed.
struct BIO;
struct X509;
struct X509_NAME;
struct X509_STORE;
struct X509_STORE_CTX;
struct X509_VERIFY_PARAM;
struct EVP_PKEY;
struct EVP_PKEY_CTX;
struct EVP_MD;
struct EVP_MD_CTX;
struct ENGINE;
struct OPENSSL_STACK;
using PasswordCallback = int (*)(char*, int, int, void*);

// clang-format off
#define TORCH_LIBCRYPTO_SYMBOLS(_)                                                                  \
  _(BIO*, BIO_new_file, const char*, const char*)                                                   \
  _(int, BIO_free, BIO*)                                                                            \
  _(X509*, PEM_read_bio_X509, BIO*, X509**, PasswordCallback, void*)                                \
  _(EVP_PKEY*, PEM_read_bio_PrivateKey, BIO*, EVP_PKEY**, PasswordCallback, void*)                  \
  _(int, i2d_X509, const X509*, unsigned char**)                                                    \
  _(X509*, d2i_X509, X509**, const unsigned char**, long)                                           \
  _(void, X509_free, X509*)                                                                         \
  _(EVP_PKEY*, X509_get0_pubkey, const X509*)                                                       \
  _(int, X509_check_private_key, const X509*, const EVP_PKEY*)                                       \
  _(X509_NAME*, X509_get_subject_name, const X509*)                                                 \
  _(char*, X509_NAME_oneline, const X509_NAME*, char*, int)                                         \
  _(void, EVP_PKEY_free, EVP_PKEY*)                                                                 \
  _(X509_STORE*, X509_STORE_new)                                                                    \
  _(void, X509_STORE_free, X509_STORE*)                                                             \
  _(int, X509_STORE_load_file, X509_STORE*, const char*)                                            \
  _(X509_STORE_CTX*, X509_STORE_CTX_new)                                                            \
  _(void, X509_STORE_CTX_free, X509_STORE_CTX*)                                                     \
  _(int, X509_STORE_CTX_init, X509_STORE_CTX*, X509_STORE*, X509*, OPENSSL_STACK*)                  \
  _(int, X509_verify_cert, X509_STORE_CTX*)                                                         \
  _(int, X509_STORE_CTX_get_error, const X509_STORE_CTX*)                                           \
  _(X509_VERIFY_PARAM*, X509_STORE_CTX_get0_param, X509_STORE_CTX*)                                 \
  _(void, X509_VERIFY_PARAM_set_auth_level, X509_VERIFY_PARAM*, int)                                \
  _(const char*, X509_verify_cert_error_string, long)                                               \
  _(OPENSSL_STACK*, OPENSSL_sk_new_null)                                                            \
  _(int, OPENSSL_sk_push, OPENSSL_STACK*, const void*)                                              \
  _(void, OPENSSL_sk_free, OPENSSL_STACK*)                                                          \
  _(EVP_MD_CTX*, EVP_MD_CTX_new)                                                                    \
  _(void, EVP_MD_CTX_free, EVP_MD_CTX*)                                                             \
  _(const EVP_MD*, EVP_sha256)                                                                      \
  _(int, EVP_DigestSignInit, EVP_MD_CTX*, EVP_PKEY_CTX**, const EVP_MD*, ENGINE*, EVP_PKEY*)        \
  _(int, EVP_DigestSign, EVP_MD_CTX*, unsigned char*, size_t*, const unsigned char*, size_t)        \
  _(int, EVP_DigestVerifyInit, EVP_MD_CTX*, EVP_PKEY_CTX**, const EVP_MD*, ENGINE*, EVP_PKEY*)      \
  _(int, EVP_DigestVerify, EVP_MD_CTX*, const unsigned char*, size_t, const unsigned char*, size_t) \
  _(int, RAND_bytes, unsigned char*, int)                                                           \
  _(unsigned long, ERR_get_error)                                                                   \
  _(unsigned long, ERR_peek_last_error)                                                             \
  _(void, ERR_error_string_n, unsigned long, char*, size_t)
// clang-format on

struct LibCrypto {
#define TORCH_LIBCRYPTO_MEMBER(ret, name, ...) ret (*name)(__VA_ARGS__){};
  TORCH_LIBCRYPTO_SYMBOLS(TORCH_LIBCRYPTO_MEMBER)
#undef TORCH_LIBCRYPTO_MEMBER

  LibCrypto() try {
    // The handle is leaked: the table is used for the lifetime of the process.
    at::DynamicLibrary lib("libcrypto.so.3", nullptr, /*leak_handle=*/true);
#define TORCH_LIBCRYPTO_LOAD(ret, name, ...) \
  name = reinterpret_cast<decltype(name)>(lib.sym(#name));
    TORCH_LIBCRYPTO_SYMBOLS(TORCH_LIBCRYPTO_LOAD)
#undef TORCH_LIBCRYPTO_LOAD
  } catch (const c10::DynamicLibraryError& e) {
    C10_THROW_ERROR(
        DistStoreError,
        std::string("TCPStore authentication requires OpenSSL 3 libcrypto: ") +
            e.what_without_backtrace());
  }
};

const LibCrypto& crypto() {
  static const LibCrypto lib;
  return lib;
}

template <typename T, auto Free>
struct CryptoDeleter {
  void operator()(T* p) const {
    (crypto().*Free)(p);
  }
};

template <typename T, auto Free>
using CryptoPtr = std::unique_ptr<T, CryptoDeleter<T, Free>>;
using BioPtr = CryptoPtr<BIO, &LibCrypto::BIO_free>;
using X509Ptr = CryptoPtr<X509, &LibCrypto::X509_free>;
using PkeyPtr = CryptoPtr<EVP_PKEY, &LibCrypto::EVP_PKEY_free>;
using StorePtr = CryptoPtr<X509_STORE, &LibCrypto::X509_STORE_free>;
using StoreCtxPtr = CryptoPtr<X509_STORE_CTX, &LibCrypto::X509_STORE_CTX_free>;
using StackPtr = CryptoPtr<OPENSSL_STACK, &LibCrypto::OPENSSL_sk_free>;
using MdCtxPtr = CryptoPtr<EVP_MD_CTX, &LibCrypto::EVP_MD_CTX_free>;

constexpr std::string_view kServerLabel = "torch-tcpstore-auth-v1 server";
constexpr std::string_view kClientLabel = "torch-tcpstore-auth-v1 client";
constexpr size_t kNonceSize = std::tuple_size_v<TCPStoreAuth::Nonce>;
// ERR_PACK(ERR_LIB_PEM, 0, PEM_R_NO_START_LINE): no further PEM block.
constexpr unsigned long kPemNoStartLine = 0x0480006C;

// Drains the thread's OpenSSL error queue.
std::string cryptoError() {
  std::string out;
  while (unsigned long err = crypto().ERR_get_error()) {
    char buf[256];
    crypto().ERR_error_string_n(err, buf, sizeof(buf));
    out += out.empty() ? "" : "; ";
    out += buf;
  }
  return out.empty() ? "unknown error" : out;
}

int noPassword(char*, int, int, void*) {
  return 0;
}

// Avoids echoing PEM contents passed by mistake instead of a path.
std::string displayPath(const std::string& path) {
  if (path.find('\n') != std::string::npos ||
      path.find("-----BEGIN") != std::string::npos) {
    return "<value is not a path>";
  }
  return path;
}

BioPtr openFile(const std::string& path) {
  BioPtr bio(crypto().BIO_new_file(path.c_str(), "r"));
  TORCH_CHECK_WITH(
      DistStoreError,
      bio,
      "Failed to open ",
      displayPath(path),
      ": ",
      cryptoError());
  return bio;
}

std::string subject(const X509* cert) {
  char buf[512];
  const char* name = crypto().X509_NAME_oneline(
      crypto().X509_get_subject_name(cert), buf, sizeof(buf));
  return name == nullptr ? "<unknown>" : name;
}

void appendU32(std::vector<uint8_t>& out, uint32_t value) {
  auto p = reinterpret_cast<const uint8_t*>(&value);
  out.insert(out.end(), p, p + sizeof(value));
}

std::vector<uint8_t> frame(const std::vector<uint8_t>& payload) {
  std::vector<uint8_t> out;
  appendU32(out, static_cast<uint32_t>(payload.size()));
  out.insert(out.end(), payload.begin(), payload.end());
  return out;
}

void randomNonce(uint8_t* out) {
  TORCH_CHECK_WITH(
      DistStoreError,
      crypto().RAND_bytes(out, kNonceSize) == 1,
      "RAND_bytes: ",
      cryptoError());
}

// Returns a SHA-256 context initialized by EVP_DigestSignInit or
// EVP_DigestVerifyInit, or nullptr.
MdCtxPtr digestCtx(
    decltype(LibCrypto::EVP_DigestSignInit) init,
    EVP_PKEY* key) {
  MdCtxPtr ctx(crypto().EVP_MD_CTX_new());
  if (ctx &&
      init(ctx.get(), nullptr, crypto().EVP_sha256(), nullptr, key) == 1) {
    return ctx;
  }
  return nullptr;
}

std::vector<uint8_t> signedData(
    std::string_view label,
    const uint8_t* clientNonce,
    const uint8_t* serverNonce) {
  std::vector<uint8_t> out(label.begin(), label.end());
  out.insert(out.end(), clientNonce, clientNonce + kNonceSize);
  out.insert(out.end(), serverNonce, serverNonce + kNonceSize);
  return out;
}

using Clock = std::chrono::steady_clock;

// Sends or receives exactly size bytes before the deadline. Transport failures
// raise DistNetworkError so that clients retry like any other connect failure.
void transfer(
    int fd,
    uint8_t* p,
    size_t size,
    bool isSend,
    Clock::time_point deadline) {
  int sendFlags = MSG_DONTWAIT;
#ifdef MSG_NOSIGNAL
  sendFlags |= MSG_NOSIGNAL;
#endif
  while (size > 0) {
    ssize_t n = isSend ? ::send(fd, p, size, sendFlags)
                       : ::recv(fd, p, size, MSG_DONTWAIT);
    if (n > 0) {
      p += n;
      size -= n;
      continue;
    }
    TORCH_CHECK_WITH(
        DistNetworkError,
        n < 0,
        "Connection closed during TCPStore authentication");
    TORCH_CHECK_WITH(
        DistNetworkError,
        errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR,
        "TCPStore authentication I/O failed: ",
        c10::utils::str_error(errno));
    auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - Clock::now());
    TORCH_CHECK_WITH(
        DistNetworkError,
        remaining.count() > 0,
        "Timed out during TCPStore authentication");
    pollfd pfd{fd, static_cast<short>(isSend ? POLLOUT : POLLIN), 0};
    // Errors and timeouts surface on the next send/recv.
    (void)::poll(
        &pfd,
        1,
        static_cast<int>(std::min<int64_t>(remaining.count(), INT_MAX)));
  }
}

} // namespace

struct TCPStoreAuth::Impl {
  std::vector<uint8_t> chain; // DER certificates, leaf first.
  PkeyPtr key;
  StorePtr ca;

  static std::unique_ptr<Impl> load(
      const std::string& certPath,
      const std::string& keyPath,
      const std::string& caPath);

  std::vector<uint8_t> sign(
      std::string_view label,
      const uint8_t* clientNonce,
      const uint8_t* serverNonce) const {
    auto data = signedData(label, clientNonce, serverNonce);
    auto ctx = digestCtx(crypto().EVP_DigestSignInit, key.get());
    size_t len = 0;
    bool ok = ctx &&
        crypto().EVP_DigestSign(
            ctx.get(), nullptr, &len, data.data(), data.size()) == 1;
    std::vector<uint8_t> sig(len);
    ok = ok &&
        crypto().EVP_DigestSign(
            ctx.get(), sig.data(), &len, data.data(), data.size()) == 1;
    TORCH_CHECK_WITH(
        DistStoreError,
        ok,
        "TCPStore authentication signing failed: ",
        cryptoError());
    sig.resize(len);
    return sig;
  }

  // Verifies the peer's DER chain against the CA and its signature with the
  // leaf key.
  void verifyPeer(
      const uint8_t* chainData,
      size_t chainSize,
      const uint8_t* sig,
      size_t sigSize,
      std::string_view label,
      const uint8_t* clientNonce,
      const uint8_t* serverNonce) const {
    std::vector<X509Ptr> peerChain;
    const uint8_t* end = chainData + chainSize;
    while (chainData < end) {
      peerChain.emplace_back(
          crypto().d2i_X509(nullptr, &chainData, end - chainData));
      TORCH_CHECK_WITH(
          DistStoreError,
          peerChain.back(),
          "Invalid peer certificate: ",
          cryptoError());
    }
    TORCH_CHECK_WITH(
        DistStoreError, !peerChain.empty(), "Peer sent no certificate");

    X509* leaf = peerChain.front().get();
    StackPtr untrusted(crypto().OPENSSL_sk_new_null());
    StoreCtxPtr storeCtx(crypto().X509_STORE_CTX_new());
    TORCH_CHECK_WITH(
        DistStoreError,
        untrusted && storeCtx &&
            crypto().X509_STORE_CTX_init(
                storeCtx.get(), ca.get(), leaf, untrusted.get()) == 1,
        "TCPStore authentication: ",
        cryptoError());
    for (size_t i = 1; i < peerChain.size(); ++i) {
      TORCH_CHECK_WITH(
          DistStoreError,
          crypto().OPENSSL_sk_push(untrusted.get(), peerChain[i].get()) > 0,
          "TCPStore authentication: ",
          cryptoError());
    }
    // Security level 2: RSA/DSA/DH keys >= 2048 bits, EC >= 224 bits and no
    // SHA-1 signatures in the chain.
    crypto().X509_VERIFY_PARAM_set_auth_level(
        crypto().X509_STORE_CTX_get0_param(storeCtx.get()), 2);
    if (crypto().X509_verify_cert(storeCtx.get()) != 1) {
      int err = crypto().X509_STORE_CTX_get_error(storeCtx.get());
      cryptoError();
      C10_THROW_ERROR(
          DistStoreError,
          "TCPStore peer certificate verification failed for " + subject(leaf) +
              ": " + crypto().X509_verify_cert_error_string(err));
    }

    auto data = signedData(label, clientNonce, serverNonce);
    auto ctx = digestCtx(
        crypto().EVP_DigestVerifyInit, crypto().X509_get0_pubkey(leaf));
    if (!ctx ||
        crypto().EVP_DigestVerify(
            ctx.get(), sig, sigSize, data.data(), data.size()) != 1) {
      cryptoError();
      C10_THROW_ERROR(
          DistStoreError,
          "TCPStore peer signature verification failed for " + subject(leaf));
    }
  }
};

std::unique_ptr<TCPStoreAuth::Impl> TCPStoreAuth::Impl::load(
    const std::string& certPath,
    const std::string& keyPath,
    const std::string& caPath) {
  auto impl = std::make_unique<Impl>();
  BioPtr certBio = openFile(certPath);
  X509Ptr leaf;
  while (X509Ptr cert{crypto().PEM_read_bio_X509(
      certBio.get(), nullptr, noPassword, nullptr)}) {
    int len = crypto().i2d_X509(cert.get(), nullptr);
    TORCH_CHECK_WITH(DistStoreError, len > 0, "i2d_X509: ", cryptoError());
    size_t offset = impl->chain.size();
    impl->chain.resize(offset + len);
    unsigned char* p = impl->chain.data() + offset;
    crypto().i2d_X509(cert.get(), &p);
    if (!leaf) {
      leaf = std::move(cert);
    }
  }
  TORCH_CHECK_WITH(
      DistStoreError,
      crypto().ERR_peek_last_error() == kPemNoStartLine,
      "Failed to read certificates from ",
      displayPath(certPath),
      ": ",
      cryptoError());
  cryptoError();
  TORCH_CHECK_WITH(
      DistStoreError, leaf, "No certificate found in ", displayPath(certPath));

  BioPtr keyBio = openFile(keyPath);
  impl->key.reset(crypto().PEM_read_bio_PrivateKey(
      keyBio.get(), nullptr, noPassword, nullptr));
  TORCH_CHECK_WITH(
      DistStoreError,
      impl->key,
      "Failed to read unencrypted private key from ",
      displayPath(keyPath),
      ": ",
      cryptoError());
  TORCH_CHECK_WITH(
      DistStoreError,
      crypto().X509_check_private_key(leaf.get(), impl->key.get()) == 1,
      "Private key ",
      displayPath(keyPath),
      " does not match certificate ",
      displayPath(certPath),
      ": ",
      cryptoError());

  impl->ca.reset(crypto().X509_STORE_new());
  TORCH_CHECK_WITH(
      DistStoreError,
      impl->ca &&
          crypto().X509_STORE_load_file(impl->ca.get(), caPath.c_str()) == 1,
      "Failed to load CA file ",
      displayPath(caPath),
      ": ",
      cryptoError());
  return impl;
}

void TCPStoreAuth::clientHandshake(int fd, Clock::time_point deadline) const {
  auto send = [&](std::vector<uint8_t> data) {
    transfer(fd, data.data(), data.size(), true, deadline);
  };
  auto recv = [&](void* data, size_t size) {
    transfer(fd, static_cast<uint8_t*>(data), size, false, deadline);
  };

  Nonce clientNonce{};
  randomNonce(clientNonce.data());
  std::vector<uint8_t> hello(clientNonce.begin(), clientNonce.end());
  hello.insert(hello.end(), impl_->chain.begin(), impl_->chain.end());
  std::vector<uint8_t> msg(kAuthMagic.begin(), kAuthMagic.end());
  auto framed = frame(hello);
  msg.insert(msg.end(), framed.begin(), framed.end());
  send(std::move(msg));

  uint32_t size = 0;
  recv(&size, sizeof(size));
  TORCH_CHECK_WITH(
      DistStoreError,
      size != kAuthMismatch,
      "TCPStore server does not have authentication enabled; disable authentication on the client or enable it on the server");
  constexpr size_t chainOffset = kNonceSize + sizeof(uint32_t);
  TORCH_CHECK_WITH(
      DistStoreError,
      size <= kAuthMaxFrameSize && size >= chainOffset,
      "Invalid TCPStore authentication reply size: ",
      size);
  std::vector<uint8_t> reply(size);
  recv(reply.data(), size);

  const uint8_t* serverNonce = reply.data();
  uint32_t chainSize = 0;
  std::memcpy(&chainSize, reply.data() + kNonceSize, sizeof(chainSize));
  TORCH_CHECK_WITH(
      DistStoreError,
      chainSize <= size - chainOffset,
      "Invalid TCPStore authentication reply");
  size_t sigOffset = chainOffset + chainSize;
  impl_->verifyPeer(
      reply.data() + chainOffset,
      chainSize,
      reply.data() + sigOffset,
      size - sigOffset,
      kServerLabel,
      clientNonce.data(),
      serverNonce);

  send(frame(impl_->sign(kClientLabel, clientNonce.data(), serverNonce)));
  uint8_t status = kAuthRejected;
  recv(&status, sizeof(status));
  TORCH_CHECK_WITH(
      DistStoreError,
      status == kAuthOk,
      "TCPStore server rejected the client certificate; see server logs");
}

std::vector<uint8_t> TCPStoreAuth::serverHello(
    const std::vector<uint8_t>& clientHello,
    Nonce& serverNonce) const {
  TORCH_CHECK_WITH(
      DistStoreError,
      validHello(clientHello),
      "Invalid TCPStore authentication hello");
  randomNonce(serverNonce.data());
  std::vector<uint8_t> reply(serverNonce.begin(), serverNonce.end());
  appendU32(reply, static_cast<uint32_t>(impl_->chain.size()));
  reply.insert(reply.end(), impl_->chain.begin(), impl_->chain.end());
  auto sig = impl_->sign(kServerLabel, clientHello.data(), serverNonce.data());
  reply.insert(reply.end(), sig.begin(), sig.end());
  return frame(reply);
}

void TCPStoreAuth::verifyClient(
    const std::vector<uint8_t>& clientHello,
    const Nonce& serverNonce,
    const std::vector<uint8_t>& clientFinish) const {
  impl_->verifyPeer(
      clientHello.data() + kNonceSize,
      clientHello.size() - kNonceSize,
      clientFinish.data(),
      clientFinish.size(),
      kClientLabel,
      clientHello.data(),
      serverNonce.data());
}

#endif // __linux__

bool TCPStoreAuth::validHello(const std::vector<uint8_t>& clientHello) {
  return clientHello.size() > std::tuple_size_v<Nonce>;
}

TCPStoreAuth::TCPStoreAuth(std::unique_ptr<Impl> impl, Paths paths)
    : impl_(std::move(impl)), paths_(std::move(paths)) {}

TCPStoreAuth::~TCPStoreAuth() = default;

std::shared_ptr<const TCPStoreAuth> TCPStoreAuth::create(
    const std::optional<std::string>& cert,
    const std::optional<std::string>& key,
    const std::optional<std::string>& ca) {
  // Explicit paths, even empty ones, take precedence over the environment as
  // a whole; the two sources are never mixed.
  bool fromEnv = !cert && !key && !ca;
  constexpr std::array<const char*, 3> kArgNames{
      "auth_cert", "auth_key", "auth_ca"};
  constexpr std::array<const char*, 3> kEnvNames{
      "TORCH_TCPSTORE_AUTH_CERT",
      "TORCH_TCPSTORE_AUTH_KEY",
      "TORCH_TCPSTORE_AUTH_CA"};
  const std::array<const std::optional<std::string>*, 3> args{&cert, &key, &ca};
  Paths paths;
  for (size_t i = 0; i < paths.size(); ++i) {
    paths[i] = fromEnv ? c10::utils::get_env(kEnvNames[i]).value_or("")
                       : args[i]->value_or("");
  }
  if (paths[0].empty() && paths[1].empty() && paths[2].empty()) {
    return nullptr;
  }
  std::string missing;
  for (size_t i = 0; i < paths.size(); ++i) {
    if (paths[i].empty()) {
      missing += missing.empty() ? "" : ", ";
      missing += fromEnv ? kEnvNames[i] : kArgNames[i];
    }
  }
  TORCH_CHECK_WITH(
      DistStoreError,
      missing.empty(),
      "TCPStore authentication requires a certificate, key and CA; missing ",
      missing,
      fromEnv ? " (from environment)" : " (from TCPStore options)");
#ifndef __linux__
  C10_THROW_ERROR(
      DistStoreError,
      "TCPStore authentication is not supported on this platform");
#else
  auto impl = Impl::load(paths[0], paths[1], paths[2]);
  // Canonical paths make sameConfig independent of the spelling.
  for (auto& path : paths) {
    std::error_code ec;
    auto resolved = std::filesystem::weakly_canonical(path, ec);
    if (!ec) {
      path = resolved.string();
    }
  }
  return std::shared_ptr<const TCPStoreAuth>(
      new TCPStoreAuth(std::move(impl), std::move(paths)));
#endif
}

bool TCPStoreAuth::sameConfig(const TCPStoreAuth* a, const TCPStoreAuth* b) {
  if (a == nullptr || b == nullptr) {
    return a == b;
  }
  return a->paths_ == b->paths_;
}

} // namespace c10d::detail
