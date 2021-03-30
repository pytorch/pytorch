#pragma once

#include <memory>
#include <thread>
#include <unordered_map>

#include <c10d/Store.hpp>

#ifdef _WIN32
#include <c10d/WinSockUtils.hpp>
#else
#include <c10d/UnixSockUtils.hpp>
#endif

namespace c10d {

class TCPStoreDaemon {
 public:
  explicit TCPStoreDaemon(int storeListenSocket);
  ~TCPStoreDaemon();

  void join();

 protected:
  void run();
  void stop();

  void queryFds(std::vector<struct pollfd>& fds);
  void query(int socket);

  void setHandler(int socket);
  void compareSetHandler(int socket);
  void addHandler(int socket);
  void getHandler(int socket) const;
  void checkHandler(int socket) const;
  void getNumKeysHandler(int socket) const;
  void deleteHandler(int socket);
  void waitHandler(int socket);

  bool checkKeys(const std::vector<std::string>& keys) const;
  void wakeupWaitingClients(const std::string& key);

  void initStopSignal();
  void closeStopSignal();

  std::thread daemonThread_;
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of sockets waiting on it
  std::unordered_map<std::string, std::vector<int>> waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<int, size_t> keysAwaited_;

  std::vector<int> sockets_;
  int storeListenSocket_;
#ifdef _WIN32
  const std::chrono::milliseconds checkTimeout_
      = std::chrono::milliseconds(10);
  HANDLE ghStopEvent_;
#else
  std::vector<int> controlPipeFd_{-1, -1};
#endif
};

class TCPStore : public Store {
 public:
  explicit TCPStore(
      const std::string& masterAddr,
      PortType masterPort,
      c10::optional<int> numWorkers = c10::nullopt_t(-1),
      bool isServer = false,
      const std::chrono::milliseconds& timeout = kDefaultTimeout,
      bool waitWorkers = true);

  virtual ~TCPStore();

  void set(const std::string& key, const std::vector<uint8_t>& value) override;

  std::vector<uint8_t> compareSet(
      const std::string& key,
      const std::vector<uint8_t>& currentValue,
      const std::vector<uint8_t>& newValue) override;

  std::vector<uint8_t> get(const std::string& key) override;

  int64_t add(const std::string& key, int64_t value) override;

  bool deleteKey(const std::string& key) override;

  bool check(const std::vector<std::string>& keys) override;

  int64_t getNumKeys() override;

  void wait(const std::vector<std::string>& keys) override;

  void wait(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout) override;

  // Waits for all workers to join.
  void waitForWorkers();

  // Returns the hostname used by the TCPStore.
  const std::string& getHost() const noexcept;

  // Returns the port used by the TCPStore.
  PortType getPort() const noexcept;

 protected:
  int64_t addHelper_(const std::string& key, int64_t value);
  std::vector<uint8_t> getHelper_(const std::string& key);
  void waitHelper_(
      const std::vector<std::string>& keys,
      const std::chrono::milliseconds& timeout);

  bool isServer_;
  int storeSocket_ = -1;
  int masterListenSocket_ = -1;

  std::string tcpStoreAddr_;
  PortType tcpStorePort_;

  c10::optional<int> numWorkers_;
  const std::string initKey_;
  const std::string regularPrefix_;

  // Only needs to be launched as the server
  std::unique_ptr<TCPStoreDaemon> tcpStoreDaemon_ = nullptr;
};

} // namespace c10d
