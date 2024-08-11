import csv
from abc import ABC, abstractmethod

import torch._C._instruction_counter as i_counter


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
            self.results.append(
                (self.name(), "instruction_count", self.count_instructions())
            )
        return self
