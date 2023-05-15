#pragma once
#include <ext/stdio_filebuf.h>
#include <sys/wait.h>
#include <torch/csrc/profiler/unwind/unwind_error.h>
#include <unistd.h>
#include <memory>

// helper to open a process with stdin/stdout/stderr streams.
struct Communicate {
  Communicate(const char* command, const char** args) {
    if (pipe(inpipe_) < 0 || pipe(outpipe_) < 0 || pipe(errpipe_) < 0) {
      throw UnwindError("pipe() failed");
    }
    pid_t pid = fork();
    if (pid < 0) {
      throw UnwindError("fork() failed");
    } else if (pid == 0) { // child process
      close(inpipe_[1]);
      close(outpipe_[0]);
      close(errpipe_[0]);

      dup2(inpipe_[0], STDIN_FILENO);
      dup2(outpipe_[1], STDOUT_FILENO);
      dup2(errpipe_[1], STDERR_FILENO);
      execvp(command, (char* const*)args);
      throw UnwindError("failed execvp");
    } else { // parent process
      close(inpipe_[0]);
      close(outpipe_[1]);
      close(errpipe_[1]);
      outbuf_.reset(
          new __gnu_cxx::stdio_filebuf<char>(inpipe_[1], std::ios::out));
      inbuf_.reset(
          new __gnu_cxx::stdio_filebuf<char>(outpipe_[0], std::ios::in));
      errbuf_.reset(
          new __gnu_cxx::stdio_filebuf<char>(errpipe_[0], std::ios::in));
      in_.reset(new std::istream(inbuf_.get()));
      out_.reset(new std::ostream(outbuf_.get()));
      err_.reset(new std::ostream(errbuf_.get()));
    }
  }
  ~Communicate() {
    close(inpipe_[1]);
    close(outpipe_[0]);
    close(errpipe_[0]);
  }
  std::ostream& out() {
    return *out_;
  }
  std::ostream& err() {
    return *err_;
  }
  std::istream& in() {
    return *in_;
  }

 private:
  int inpipe_[2];
  int outpipe_[2];
  int errpipe_[2];
  std::unique_ptr<__gnu_cxx::stdio_filebuf<char>> outbuf_, inbuf_, errbuf_;
  std::unique_ptr<std::istream> in_;
  std::unique_ptr<std::ostream> out_;
  std::unique_ptr<std::ostream> err_;
};
