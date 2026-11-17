#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  static const char message[] = "release-tls-native-marker-20261117\n";
  const char* path = getenv("RLS05_MARKER");
  if (path == NULL) {
    return 72;
  }
  FILE* stream = fopen(path, "wb");
  if (stream == NULL) {
    return 72;
  }
  int written = fwrite(message, 1, sizeof(message) - 1, stream) == sizeof(message) - 1;
  int closed = fclose(stream) == 0;
  return written && closed ? 73 : 72;
}
