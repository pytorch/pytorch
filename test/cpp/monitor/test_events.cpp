#include <gtest/gtest.h>

#include <torch/csrc/monitor/events.h>
#include <torch/torch.h>

using namespace torch::monitor;

struct AggregatingEventHandler : public EventHandler {
  std::vector<Event> events;

  void handle(const Event& e) override {
    events.emplace_back(e);
  }
};

TEST(EventsTest, EventHandler) {
  Event e;
  e.name = "test";
  e.timestamp = std::chrono::system_clock::now();
  e.data["string"] = "asdf";
  e.data["double"] = 1234.5678;
  e.data["int"] = 1234L;
  e.data["bool"] = true;
  e.data["tensor"] = torch::ones(5);

  // log to nothing
  logEvent(e);

  auto handler = std::make_shared<AggregatingEventHandler>();
  registerEventHandler(handler);

  logEvent(e);
  ASSERT_EQ(handler->events.size(), 1);
  ASSERT_EQ(e, handler->events.at(0));

  unregisterEventHandler(handler);
  logEvent(e);
  // handler unregister, didn't log
  ASSERT_EQ(handler->events.size(), 1);
}
