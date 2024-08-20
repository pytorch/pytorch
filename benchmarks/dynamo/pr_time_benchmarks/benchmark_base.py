import csv
from abc import ABC, abstractmethod
from fbscribelogger import make_scribe_logger

import torch._C._instruction_counter as i_counter


scribe_log_torch_benchmark_compile_time = make_scribe_logger(
    "TorchBenchmarkCompileTime",
    """
# Use this struct to interact with Scribe APIs in order to Log data.
struct TorchBenchmarkCompileTimeLogEntry {

  # The commit SHA that triggered the workflow. Derived from GITHUB_SHA.
  4: optional string commit_sha;

  # The job_id of the current job. Derived from GITHUB_JOB.
  5: optional string job_name;

  # The unit timestamp in second for the Scuba Time Column override
  6: optional i64 time;
  7: optional i64 instruction_count; # Instruction count of compilation step
  8: optional string name; # Benchmark name

  # Docker image the job was executed on, including hash.  Specific to pytorch/pytorch.  Derived from BRANCH.
  9: optional string docker_image;

  # Branch the job was executed on, e.g., main.  Specific to pytorch/pytorch.  Derived from BRANCH.
  10: optional string branch;

  # The operating system of the runner executing the job. Derived from RUNNER_OS.
  11: optional string runner_os;

  # The name of the runner executing the job. Derived from RUNNER_NAME.
  12: optional string runner_name;

  # The architecture of the runner executing the job. Derived from RUNNER_ARCH.
  13: optional string runner_arch;

  # A unique number for each run of a particular workflow in a repository. Derived from GITHUB_RUN_NUMBER.
  14: optional i64 github_run_number;

  # Commit date (not author date) of the commit in commit_sha as timestamp.  Increasing if merge bot is used, though not monotonic; duplicates occur when stack is landed.
  16: optional i64 commit_date;

  # A unique number for each workflow run within a repository. Derived from GITHUB_RUN_ID.
  17: optional string github_run_id;

  # A unique number for each attempt of a particular workflow run in a repository. Derived from GITHUB_RUN_ATTEMPT.
  18: optional string github_run_attempt;

  # The owner and repository name. Derived from GITHUB_REPOSITORY.
  19: optional string github_repository;

  # Indicates if branch protections or rulesets are configured for the ref that triggered the workflow run. Derived from GITHUB_REF_PROTECTED.
  20: optional bool github_ref_protected;

  # The fully-formed ref of the branch or tag that triggered the workflow run. Derived from GITHUB_REF.
  21: optional string github_ref;

  # The name of the base ref or target branch of the pull request in a workflow run. Derived from GITHUB_BASE_REF.
  22: optional string github_base_ref;

  # The name of the action currently running, or the id of a step. Derived from GITHUB_ACTION.
  23: optional string github_action;

  # GitHub username of author of commit. Public data.
  24: optional string commit_author;

  # The weight of the record according to current sampling rate
  25: optional i64 weight;
}
""",  # noqa: B950
)


class BenchmarkBase(ABC):
    _instruction_count = False

    def enable_instruction_count(self):
        self._instruction_count = True
        return self

    def name(self):
        return ""

    def description(self):
        return ""

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def work(self):
        pass

    def prepare_once(self):  # noqa: B027
        pass

    def count_instructions(self):
        print(f"collecting instruction count for {self.name()}")
        self.prepare_once()

        results = []
        for i in range(10):
            self.prepare()
            id = i_counter.start()
            self.work()
            count = i_counter.end(id)
            print(f"instruction count for iteration {i} is {count}")
            if i != 0:
                results.append(count)
        return min(results)

    def append_results(self, path):
        with open(path, "a", newline="") as csvfile:
            # Create a writer object
            writer = csv.writer(csvfile)
            # Write the data to the CSV file
            for entry in self.results:
                writer.writerow(entry)

    def print(self):
        for entry in self.results:
            print(f"{entry[0]},{entry[1]},{entry[2]}")

    def collect_all(self):
        self.results = []
        if self._instruction_count:
            r = self.count_instructions()
            self.results.append(
                (self.name(), "instruction_count", r)
            )
            scribe_log_torch_benchmark_compile_time(
                name=self.name(),
                instruction_count=r,
            )
        return self
