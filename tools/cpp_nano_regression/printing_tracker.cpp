#include <cstdio>

extern "C" {
void nano_tracking_log(const char* event) {
  printf("NANO TRACKER %s\n", event);
}
}
