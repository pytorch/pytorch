#include <gtest/gtest.h>

#include <torch/csrc/distributed/rpc/process_group_agent.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/lib/c10d/HashStore.hpp>
#include <torch/lib/c10d/ProcessGroupGloo.hpp>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace torch::distributed::rpc;

namespace {
std::vector<char> toVec(const std::string& s) {
  std::vector<char> res(s.begin(), s.end());
  return res;
}

class Baton {
 public:
  void post() {
    std::unique_lock<std::mutex> l(lock);
    done = true;
    cv.notify_all();
  }
  void wait() {
    std::unique_lock<std::mutex> l(lock);
    cv.wait(l, [&]() { return done; });
  }
  std::mutex lock;
  std::condition_variable cv;
  bool done{false};
};

constexpr auto kRpcTimeout = std::chrono::milliseconds(250);
constexpr auto kMinimalTime = std::chrono::milliseconds(10);
}; // namespace

class EchoingRequestCallback : public RequestCallback {
 public:
  std::shared_ptr<FutureMessage> processMessage(
      Message& request) const override {
    auto id = request.id();
    auto payload = std::move(request).movePayload();
    auto tensors = std::move(request).moveTensors();
    std::string payloadText(payload.begin(), payload.end());
    Message response(std::move(payload), std::move(tensors), SCRIPT_RET, id);

    // This handler typically echoes, but responds to various commands in the
    // payload.
    if (payloadText.find("completedInFuture") != std::string::npos) {
      auto future = std::make_shared<FutureMessage>();
      std::thread t([response, future]() {
        std::this_thread::sleep_for(kMinimalTime); /* sleep override */
        future->markCompleted(response);
      });
      {
        std::lock_guard<std::mutex> l(finishingThreadsLock_);
        finishingThreads_.emplace_back(std::move(t));
      }
      return future;
    }
    if (payloadText.find("futureIsError") != std::string::npos) {
      auto ret = std::make_shared<FutureMessage>();
      ret->setError(payloadText);
      return ret;
    }
    if (payloadText.find("operatorThrows") != std::string::npos) {
      throw std::runtime_error("bye");
    }
    if (payloadText.find("sleep") != std::string::npos) {
      // Respond but sleep first to trigger timeout.
      std::this_thread::sleep_for(kRpcTimeout * 2); /* sleep override */
    }
    return std::make_shared<FutureMessage>(std::move(response));
  }

 private:
  mutable std::mutex finishingThreadsLock_;
  mutable std::vector<std::thread> finishingThreads_;
};

class EchoingProcessGroupAgent : public ProcessGroupAgent {
 public:
  EchoingProcessGroupAgent(
      std::string workerName,
      std::shared_ptr<c10d::ProcessGroup> pg,
      int numSendRecvThreads,
      std::chrono::milliseconds rpcTimeout)
      : ProcessGroupAgent(workerName, pg, numSendRecvThreads, rpcTimeout) {
    // Minor hack to base class.
    const_cast<std::unique_ptr<RequestCallback>*>(&cb_)->reset(
        new EchoingRequestCallback);
  }
};

class ProcessGroupHelper {
 public:
  static ProcessGroupHelper* getSingleton() {
    static std::once_flag once;
    // explicitly leak on exit, avoid destructor issues.
    static ProcessGroupHelper* singleton;
    std::call_once(once, [&]() {
      torch::distributed::autograd::DistAutogradContainer::init(0);
      singleton = new ProcessGroupHelper(2);
    });
    return singleton;
  }

  explicit ProcessGroupHelper(int numworkers) {
    queues_.resize(numworkers);
    for (int i = 0; i < numworkers; ++i) {
      queues_[i].reset(new WorkQueue);
      threads_.push_back(std::thread([i, this, numworkers]() {
        ::c10d::ProcessGroupGloo::Options options;
        options.timeout = std::chrono::milliseconds(1000);
        options.devices.push_back(
            ::c10d::ProcessGroupGloo::createDeviceForHostname("127.0.0.1"));

        auto pg = std::make_shared<::c10d::ProcessGroupGloo>(
            store_, i, numworkers, options);
        constexpr uint32_t kNumThreads = 4;
        auto pga = std::make_shared<EchoingProcessGroupAgent>(
            "worker" + std::to_string(i), pg, kNumThreads, kRpcTimeout);
        pga->start();
        WorkQueue& q = *queues_[i];
        for (;;) {
          std::unique_lock<std::mutex> l(q.lock);
          q.cv.wait(l, [&]() { return !q.queue.empty(); });
          std::function<bool(RpcAgent&)> item = std::move(q.queue.front());
          q.queue.pop_front();
          if (!item(*pga))
            break;
        }
      }));
    }
  }

  ~ProcessGroupHelper() {
    for (auto& t : threads_) {
      t.join();
    }
  }

  void queueWork(int num, std::function<bool(RpcAgent&)> work) {
    WorkQueue& q = *queues_[num];
    std::unique_lock<std::mutex> l(q.lock);
    q.queue.push_back(std::move(work));
    q.cv.notify_all();
  }

  void runSendWorkCase(
      Message toSend,
      const std::function<void(std::shared_ptr<FutureMessage>)>& func) {
    Baton b;
    queueWork(0, [&](RpcAgent& agent) -> bool {
      std::shared_ptr<FutureMessage> m =
          agent.send(agent.getWorkerInfo(1), std::move(toSend));
      func(m);
      b.post();
      return true;
    });
    b.wait();
  }

  int64_t nextId() {
    return nextId_++;
  }

 private:
  std::shared_ptr<::c10d::Store> store_{std::make_shared<::c10d::HashStore>()};
  std::vector<std::thread> threads_;
  std::atomic<int64_t> nextId_{0};

  struct WorkQueue {
    std::mutex lock;
    std::condition_variable cv;
    std::deque<std::function<bool(RpcAgent&)>> queue;
  };
  std::vector<std::unique_ptr<WorkQueue>> queues_;
};

TEST(ProcessGroupAgent, BasicEcho) {
  ProcessGroupHelper* helper = ProcessGroupHelper::getSingleton();
  auto tensor = torch::randn({5});
  Message toSend(toVec("hi"), {tensor}, SCRIPT_REMOTE_CALL, helper->nextId());
  helper->runSendWorkCase(
      std::move(toSend), [&](std::shared_ptr<FutureMessage> responseFuture) {
        auto result = responseFuture->wait();
        EXPECT_EQ(result.payload(), toVec("hi"));
        EXPECT_TRUE(torch::equal(result.tensors()[0], tensor));
      });
}

TEST(ProcessGroupAgent, CompletedInFuture) {
  ProcessGroupHelper* helper = ProcessGroupHelper::getSingleton();
  auto tensor = torch::randn({5});
  Message toSend(
      toVec("completedInFuture"),
      {tensor},
      SCRIPT_REMOTE_CALL,
      helper->nextId());
  helper->runSendWorkCase(
      std::move(toSend), [&](std::shared_ptr<FutureMessage> responseFuture) {
        auto response = responseFuture->wait();
        EXPECT_EQ(response.payload(), toVec("completedInFuture"));
        EXPECT_TRUE(torch::equal(response.tensors()[0], tensor));
      });
}

TEST(ProcessGroupAgent, OperatorThrows) {
  ProcessGroupHelper* helper = ProcessGroupHelper::getSingleton();
  auto tensor = torch::randn({5});
  Message toSend(
      toVec("operatorThrows"), {tensor}, SCRIPT_REMOTE_CALL, helper->nextId());
  helper->runSendWorkCase(
      std::move(toSend), [&](std::shared_ptr<FutureMessage> responseFuture) {
        responseFuture->waitNoThrow();
        EXPECT_TRUE(responseFuture->hasError());
      });
}

TEST(ProcessGroupAgent, FutureIsError) {
  ProcessGroupHelper* helper = ProcessGroupHelper::getSingleton();
  auto tensor = torch::randn({5});
  Message toSend(
      toVec("futureIsError"), {tensor}, SCRIPT_REMOTE_CALL, helper->nextId());
  helper->runSendWorkCase(
      std::move(toSend), [&](std::shared_ptr<FutureMessage> responseFuture) {
        responseFuture->waitNoThrow();
        EXPECT_TRUE(responseFuture->hasError());
      });
}

TEST(ProcessGroupAgent, Timeout) {
  ProcessGroupHelper* helper = ProcessGroupHelper::getSingleton();
  auto tensor = torch::randn({5});
  Message toSend(
      toVec("sleep"), {tensor}, SCRIPT_REMOTE_CALL, helper->nextId());
  helper->runSendWorkCase(
      std::move(toSend), [&](std::shared_ptr<FutureMessage> responseFuture) {
        responseFuture->waitNoThrow();
        EXPECT_TRUE(responseFuture->hasError());
      });
}
