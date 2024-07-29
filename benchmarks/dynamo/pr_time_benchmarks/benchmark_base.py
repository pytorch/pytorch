from abc import ABC, abstractmethod

from cirron import Collector


class Benchmark(ABC):
    _instruction_count = False

    def enable_instruction_count(self):
        self._instruction_count = True
        return self

    def name(self):
        return ""

    def description(self):
        return ""

    def reset(self):  # noqa: B027
        pass

    @abstractmethod
    def prepare(self):
        pass

    @abstractmethod
    def work(self):
        pass

    def count_instructions(self):
        self.reset()
        self.prepare()
        results = []
        for i in range(10):
            self.reset()
            self.prepare()
            with Collector() as collector:
                self.work()

            if i != 0:
                results.append(collector.counters.instruction_count)
        return min(results)

    def collect_all(self):
        result = []
        if self._instruction_count:
            result.append((self.name(), "instruction_count", self.count_instructions()))
        return result
