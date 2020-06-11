#include <iostream>
#include <unordered_map>

using namespace std;

// map is ok - perf doesn't really matter :)
unordered_map<string, int> events;

extern "C" {
void nano_tracking_log(const char* event) {
  ++events[string(event)];
}

void nano_tracking_dump() {
  cout << "NANO TRACKER {" << endl;
  for (const auto& it : events) {
    cout << "  " << it.first << ": " << it.second << endl;
  }
  cout << "}" << endl;
}

void nano_tracking_reset() {
  events.clear();
}

int nano_tracking_get_event(const char* event) {
  auto it = events.find(string(event));
  return it == events.end() ? 0 : it->second;
}
}

struct Finalizer {
  ~Finalizer() {
    nano_tracking_dump();
  }
};

static Finalizer _instance;
