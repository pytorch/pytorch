#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <c10/util/irange.h>
#include <fmt/format.h>

#define ERROR(msg_fmt, ...) \
  throw std::runtime_error(fmt::format(msg_fmt, ##__VA_ARGS__))

#define CHECK(cond, fmt, ...)  \
  if (!(cond)) {               \
    ERROR(fmt, ##__VA_ARGS__); \
  }

// NOLINTNEXTLINE
int main(int argc, const char** argv) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0] << " <input_library> <result_library>\n";
    return 1;
  }
  const char* filename = argv[1];
  int fd_ = open(filename, O_RDWR);
  CHECK(fd_ != -1, "failed to open {}: {}", filename, strerror(errno));
  struct stat s = {0};
  if (-1 == fstat(fd_, &s)) {
    close(fd_); // destructors don't run during exceptions
    ERROR("failed to stat {}: {}", filename, strerror(errno));
  }
  size_t n_bytes = s.st_size;
  void* mem =
      mmap(nullptr, n_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd_, 0);
  if (MAP_FAILED == mem) {
    close(fd_);
    ERROR("failed to mmap {}: {}", filename, strerror(errno));
  }

  char* data = (char*)mem;
  auto header = (Elf64_Ehdr*)data;
  auto program_headers = (Elf64_Phdr*)(data + header->e_phoff);
  auto n_program_headers = header->e_phnum;
  Elf64_Dyn* dynamic = nullptr;
  for (const auto i : c10::irange(n_program_headers)) {
    const Elf64_Phdr* phdr = &program_headers[i];
    if (phdr->p_type == PT_DYNAMIC) {
      dynamic = reinterpret_cast<Elf64_Dyn*>(data + phdr->p_offset);
      break;
    }
  }
  CHECK(
      dynamic,
      "{}: could not load dynamic section for looking up DT_NEEDED",
      filename);
  std::vector<Elf64_Dyn> entries;
  for (const Elf64_Dyn* d = dynamic; d->d_tag != DT_NULL; ++d) {
    entries.push_back(*d);
  }
  Elf64_Dyn* w = dynamic;
  for (const Elf64_Dyn& e : entries) {
    if (e.d_tag != DT_NEEDED) {
      *w++ = e;
    }
  }
  auto nwritten = w - dynamic;
  memset(w, 0, sizeof(Elf64_Dyn) * (entries.size() - nwritten));

  FILE* dst = fopen(argv[2], "w");
  CHECK(dst != nullptr, "{}: {}", argv[2], strerror(errno));
  fwrite(mem, n_bytes, 1, dst);
  fclose(dst);
  munmap(mem, n_bytes);
  close(fd_);
  return 0;
}
