#pragma once

#include <unistd.h>

struct AllocInfo {
  pid_t pid;
  char free;
  char shm;
  char filename[60];
};
