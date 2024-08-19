import csv
from abc import ABC, abstractmethod

import torch._C._instruction_counter as i_counter
import torch._dynamo.config as config
from torch._dynamo.utils import CompileTimeInstructionCounter


class BenchmarkBase(ABC):
    _instruction_count = False
    _compile_time_instruction_count = False

    def enable_instruction_count(self):
        self._instruction_count = True
        return self

    def enable_compile_time_instruction_count(self):
        self._compile_time_instruction_count = True
        return self

    def name(self):
        return ""

    def description(self):
        return ""

    @abstractmethod
    def _prepare(self):
        pass

    @abstractmethod
    def _work(self):
        pass

    def _prepare_once(self):  # noqa: B027
        pass

    def _count_instructions(self):
        print(f"collecting instruction count for {self.name()}")
        results = []
        for i in range(10):
            self._prepare()
            id = i_counter.start()
            self._work()
            count = i_counter.end(id)
            print(f"instruction count for iteration {i} is {count}")
            if i != 0:
                results.append(count)
        return min(results)

    def _count_compile_time_instructions(self):
        print(f"collecting compile time instruction count for {self.name()}")
        config.record_compile_time_instruction_count = True

        results = []
        for i in range(10):
            self._prepare()
            CompileTimeInstructionCounter.clear()
            self._work()
            count = CompileTimeInstructionCounter.value()
            print(f"compile time instruction count for iteration {i} is {count}")
            if i != 0:
                results.append(count)

        config.record_compile_time_instruction_count = False
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
        self._prepare_once()
        self.results = []
        if self._instruction_count:
            self.results.append(
                (self.name(), "instruction_count", self._count_instructions())
            )
        if self._compile_time_instruction_count:
            self.results.append(
                (
                    self.name(),
                    "compile_time_instruction_count",
                    self._count_compile_time_instructions(),
                )
            )
        return self
