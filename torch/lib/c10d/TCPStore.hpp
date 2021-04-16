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

enum class WatchResponseType : uint8_t {
  KEY_UPDATED,
  KEY_CREATED,
  KEY_DELETED,
  KEY_CALLBACK_REGISTERED
};

// Abstract base class to handle thread state for TCPStoreMasterDaemon and
// TCPStoreWorkerDaemon. Contains the windows/unix implementations to signal a
// shutdown sequence for the thread
class BackgroundThread {
 public:
  explicit BackgroundThread(int storeListenSocket);
  virtual ~BackgroundThread() = 0;

 protected:
  std::thread daemonThread_;
  int storeListenSocket_;
  std::vector<int> sockets_;
#ifdef _WIN32
  const std::chrono::milliseconds checkTimeout_ = std::chrono::milliseconds(10);
  HANDLE ghStopEvent_;
#else
  std::vector<int> controlPipeFd_{-1, -1};
#endif
 private:
  // Initialization for shutdown signal
  void initStopSignal();
  // Triggers the shutdown signal
  void stop();
  // Joins the thread
  void join();
  // Clean up the shutdown signal
  void closeStopSignal();
};

// Separate thread that is only launched on master
class TCPStoreMasterDaemon : public BackgroundThread {
 public:
  explicit TCPStoreMasterDaemon(int storeListenSocket);

 protected:
  void run();
  void queryFds(std::vector<struct pollfd>& fds);
  void query(int socket);

  // The master runs on a single thread so only
  // one handler can be executed at a time
  void setHandler(int socket);
  void compareSetHandler(int socket);
  void addHandler(int socket);
  void getHandler(int socket) const;
  void checkHandler(int socket) const;
  void getNumKeysHandler(int socket) const;
  void deleteHandler(int socket);
  void waitHandler(int socket);
  void watchHandler(int socket);

  bool checkKeys(const std::vector<std::string>& keys) const;
  // Helper function to alerts waiting workers, used in setHandler, getHandler
  void wakeupWaitingClients(const std::string& key);
  // Helper function used when the key is changed
  // used in setHandler, addHandler, getHandler, deleteHandler
  void sendKeyUpdatesToClients(
      const std::string& key,
      const enum WatchResponseType& type,
      std::vector<uint8_t>& oldData,
      std::vector<uint8_t>& newData);

 private:
  std::unordered_map<std::string, std::vector<uint8_t>> tcpStore_;
  // From key -> the list of sockets waiting on the key
  std::unordered_map<std::string, std::vector<int>> waitingSockets_;
  // From socket -> number of keys awaited
  std::unordered_map<int, size_t> keysAwaited_;
  // From key -> the list of sockets watching the key
  std::unordered_map<std::string, std::vector<int>> watchedSockets_;
};

// Separate thread that is launched on all instances (including master)
// Right now only handles callbacks registered from watchKey()
class TCPStoreWorkerDaemon : public BackgroundThread {
 public:
  explicit TCPStoreWorkerDaemon(int listenSocket);
  // Adds a callback to run key change
  void addCallback(
      std::string key,
      std::function<
          void(c10::optional<std::string>, c10::optional<std::string>)> cb);
  std::mutex mtx;
  std::condition_variable cv;
  bool callbackRegistered;

 protected:
  void run();
  void callbackHandler(int socket);
  // List of callbacks map each watched key
  std::unordered_map<
      std::string,
      std::function<
          void(c10::optional<std::string>, c10::optional<std::string>)>>
      keyToCallbacks;

 private:
  std::mutex keyToCallbacksMutex;
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

  // callback function will be given arguments (optiona<string> oldValue,
  // optional<string> newValue)
  // NOTE: calling other TCPStore APIs inside the callback is NOT threadsafe
  void watchKey(
      const std::string& key,
      std::function<
          void(c10::optional<std::string>, c10::optional<std::string>)>
          callback) override;

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
  int listenSocket_ = -1;
  int masterListenSocket_ = -1;

  std::string tcpStoreAddr_;
  PortType tcpStorePort_;

  c10::optional<int> numWorkers_;
  const std::string initKey_;
  const std::string regularPrefix_;

  std::unique_ptr<TCPStoreMasterDaemon> tcpStoreMasterDaemon_ = nullptr;
  std::unique_ptr<TCPStoreWorkerDaemon> tcpStoreWorkerDaemon_ = nullptr;
};

} // namespace c10d
