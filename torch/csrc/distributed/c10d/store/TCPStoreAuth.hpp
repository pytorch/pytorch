#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace c10d::detail {

// Optional mutual X.509 authentication of TCPStore connections, configured by
// TCPStoreOptions::auth{Cert,Key,Ca} or TORCH_TCPSTORE_AUTH_{CERT,KEY,CA}.
// Only the libuv server backend supports it.
// Each connection performs a one-time challenge-response handshake and then
// continues with the regular protocol in plaintext. This authenticates peers
// but provides no confidentiality or integrity after the handshake.
//
// Wire format (native byte order, like the rest of the protocol), where
// frame(x) = u32 size || x and chains are DER certificates, leaf first:
//   client: kAuthMagic || frame(Nc || client chain)
//   server: frame(Ns || u32 chain size || server chain || server signature)
//   client: frame(client signature)
//   server: kAuthOk or kAuthRejected
// Signatures cover a role label || Nc || Ns.

// The leading 0xFF is not a valid QueryType.
inline constexpr std::array<uint8_t, 8> kAuthMagic =
    {0xFF, 'T', 'S', 'A', 'U', 'T', 'H', 1};
// Sent by a server instead of a reply when exactly one side has
// authentication enabled. Never a valid frame size or ping nonce.
inline constexpr uint32_t kAuthMismatch = 0xFFA07B5E;
inline constexpr uint32_t kAuthMaxFrameSize = 64 * 1024;
inline constexpr uint8_t kAuthOk = 1;
inline constexpr uint8_t kAuthRejected = 0;

class TCPStoreAuth {
 public:
  using Nonce = std::array<uint8_t, 32>;

  // Loads PEM files. If no path is given, all three are read from
  // TORCH_TCPSTORE_AUTH_{CERT,KEY,CA}; otherwise the environment is ignored.
  // Returns nullptr when all paths are empty.
  static std::shared_ptr<const TCPStoreAuth> create(
      const std::optional<std::string>& cert,
      const std::optional<std::string>& key,
      const std::optional<std::string>& ca);

  // Whether a and b were created from the same canonical paths.
  static bool sameConfig(const TCPStoreAuth* a, const TCPStoreAuth* b);

  TCPStoreAuth(const TCPStoreAuth&) = delete;
  TCPStoreAuth& operator=(const TCPStoreAuth&) = delete;
  ~TCPStoreAuth();

  // Blocking handshake on a connected client socket.
  void clientHandshake(int fd, std::chrono::steady_clock::time_point deadline)
      const;

  // Non-blocking server steps. Take frame payloads; serverHello returns the
  // framed reply.
  static bool validHello(const std::vector<uint8_t>& clientHello);
  // Requires validHello(clientHello).
  std::vector<uint8_t> serverHello(
      const std::vector<uint8_t>& clientHello,
      Nonce& serverNonce) const;
  // Requires a clientHello accepted by serverHello.
  void verifyClient(
      const std::vector<uint8_t>& clientHello,
      const Nonce& serverNonce,
      const std::vector<uint8_t>& clientFinish) const;

 private:
  struct Impl;

  using Paths = std::array<std::string, 3>;

  TCPStoreAuth(std::unique_ptr<Impl> impl, Paths paths);

  std::unique_ptr<Impl> impl_;
  Paths paths_;
};

} // namespace c10d::detail
