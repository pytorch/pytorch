#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <torch/csrc/distributed/c10d/TCPStore.hpp>

// TLS with mutual certificate or pre-shared key authentication for the
// TCPStore client and the libuv server. OpenSSL (libssl.so) is loaded at
// runtime only when TLS is enabled, so there is no build-time dependency.

namespace c10d::detail {

// Content type of the first record a TLS client sends (ClientHello).
constexpr uint8_t kTLSHandshakeRecordType = 0x16;

// Fatal "protocol_version" alert record. A plaintext server sends it to a TLS
// client so the handshake fails promptly instead of timing out.
constexpr std::array<uint8_t, 7> kTLSProtocolVersionAlert =
    {0x15, 0x03, 0x03, 0x00, 0x02, 0x02, 0x46};

// Sent by a TLS server to a plaintext client in place of the ping response.
// Larger than any pid so it cannot match a ping nonce.
constexpr uint32_t tlsRequiredMagicNumber = 0x7F4C5352;

struct TLSConfig {
  std::string cert;
  std::string key;
  std::string ca;
  std::string psk; // Secret key value, never a filename or diagnostic argument.
  bool fromEnv = false;

  bool enabled() const {
    return !cert.empty() || !key.empty() || !ca.empty() || !psk.empty();
  }

  bool sameSettings(const TLSConfig& other) const {
    return cert == other.cert && key == other.key && ca == other.ca &&
        psk == other.psk;
  }
};

// The TLS options if any is set, otherwise the TORCH_TCPSTORE_TLS_CERT,
// TORCH_TCPSTORE_TLS_KEY, TORCH_TCPSTORE_TLS_CA and TORCH_TCPSTORE_TLS_PSK
// environment variables. Throws if certificate mode is missing a value or is
// combined with a PSK.
TLSConfig tlsConfig(const TCPStoreOptions& opts);

class TLSContext {
 public:
  // Returns nullptr when TLS is disabled.
  static std::shared_ptr<TLSContext> fromConfig(
      const TLSConfig& config,
      bool isServer);

  TLSContext(void* ctx, bool isServer) : ctx_{ctx}, isServer_{isServer} {}
  TLSContext(const TLSContext&) = delete;
  TLSContext& operator=(const TLSContext&) = delete;
  ~TLSContext();

  void* get() const {
    return ctx_;
  }

  bool isServer() const {
    return isServer_;
  }

  // The pre-shared key, empty in certificate mode.
  const std::string& psk() const {
    return psk_;
  }

 private:
  void* ctx_;
  bool isServer_;
  std::string psk_;
};

// A TLS connection over memory buffers. Ciphertext received from the peer is
// passed to feed() and ciphertext to send is appended to an output buffer, so
// it works with both blocking sockets and libuv.
class TLSSession {
 public:
  using Deadline = std::chrono::steady_clock::time_point;

  explicit TLSSession(std::shared_ptr<TLSContext> ctx);
  TLSSession(const TLSSession&) = delete;
  TLSSession& operator=(const TLSSession&) = delete;
  ~TLSSession();

  void feed(const void* data, size_t length);
  // Returns true once the handshake has completed.
  bool handshake();
  bool handshakeComplete() const;
  // Returns plaintext bytes read, 0 for more input, or nullopt for clean EOF.
  std::optional<size_t> read(void* data, size_t length);
  // Queues close_notify after the peer has finished sending.
  void shutdown();
  // Encrypts data and appends the records to out.
  void write(const void* data, size_t length, std::vector<uint8_t>& out);
  // Appends pending output, e.g. handshake messages or alerts, to out.
  void takeOutput(std::vector<uint8_t>& out);

  // Blocking helpers for a connected socket. No deadline means no timeout.
  void handshake(int socket, std::optional<Deadline> deadline);
  // Returns false if no plaintext is available before the deadline.
  bool waitForInput(int socket, Deadline deadline);
  void sendBytes(int socket, const void* data, size_t length, bool moreData);
  void recvBytes(int socket, void* data, size_t length);

 private:
  void flush(int socket, bool moreData = false);
  // Returns false if the deadline passed before any data arrived.
  bool recvSome(int socket, std::optional<Deadline> deadline = std::nullopt);
  [[noreturn]] void throwError(int ret, const char* op);

  std::shared_ptr<TLSContext> ctx_;
  void* ssl_;
  void* rbio_;
  void* wbio_;
};

// A socket that optionally carries TLS, for use with the tcputil helpers.
struct SocketIO {
  int socket;
  TLSSession* tls;

  void sendBytes(const void* data, size_t length, bool moreData) const;
  void recvBytes(void* data, size_t length) const;
};

} // namespace c10d::detail
