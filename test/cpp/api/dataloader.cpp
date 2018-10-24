#include <gtest/gtest.h>

#include <torch/data.h>
#include <torch/data/detail/sequencers.h>
#include <torch/tensor.h>

#include <test/cpp/api/support.h>

#include <ATen/core/ArrayRef.h>

#include <chrono>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace torch::data; // NOLINT

const std::chrono::milliseconds kMillisecond(1);

struct DummyDataset : datasets::Dataset<DummyDataset, int> {
  int get(size_t index) override {
    return 1 + index;
  }
  size_t size() const override {
    return 100;
  }
};

TEST(DataTest, DatasetCallsGetCorrectly) {
  DummyDataset d;
  std::vector<int> batch = d.get_batch({0, 1, 2, 3, 4});
  std::vector<int> expected = {1, 2, 3, 4, 5};
  ASSERT_EQ(batch, expected);
}

TEST(DataTest, TransformCallsGetApplyCorrectly) {
  struct T : transforms::Transform<int, std::string> {
    std::string apply(int input) override {
      return std::to_string(input);
    }
  };

  auto d = DummyDataset{}.map(T{});
  std::vector<std::string> batch = d.get_batch({0, 1, 2, 3, 4});
  std::vector<std::string> expected = {"1", "2", "3", "4", "5"};
  ASSERT_EQ(batch, expected);
}

struct InfiniteStreamDataset
    : datasets::BatchDataset<InfiniteStreamDataset, std::vector<int>> {
  std::vector<int> get_batch(torch::ArrayRef<size_t> batch_size) override {
    AT_ASSERT(batch_size.size() == 1);
    std::vector<int> batch(batch_size.front());
    for (auto& i : batch) {
      i = counter++;
    }
    return batch;
  }

  size_t size() const override {
    return std::numeric_limits<size_t>::max();
  }

  size_t counter = 0;
};

struct BatchSizeSampler : samplers::Sampler {
  void reset() override {}
  at::optional<std::vector<size_t>> next(size_t batch_size) override {
    return {{batch_size}};
  }
};

TEST(DataTest, InfiniteStreamDataset) {
  const size_t kBatchSize = 13;

  {
    BatchSizeSampler sampler;
    ASSERT_EQ(sampler.next(kBatchSize).value().size(), 1);
    ASSERT_EQ(sampler.next(kBatchSize).value().front(), kBatchSize);
  }

  auto dataset = InfiniteStreamDataset().map(
      transforms::Lambda<int>([](int x) { return x + 1; }));

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      DataLoaderOptions().batch_size(kBatchSize),
      BatchSizeSampler{});

  auto iterator = data_loader->begin();
  for (size_t i = 0; i < 3; ++i, ++iterator) {
    ASSERT_NE(iterator, data_loader->end());
    std::vector<int> batch = *iterator;
    ASSERT_EQ(batch.size(), kBatchSize);
    for (size_t j = 0; j < kBatchSize; ++j) {
      ASSERT_EQ(batch.at(j), 1 + (i * kBatchSize) + j);
    }
  }
}

TEST(DataTest, NoSequencerIsIdentity) {
  using namespace torch::data::detail::sequencers; // NOLINT
  NoSequencer<int> no_sequencer;
  const auto value = no_sequencer.next([] { return 5; }).value();
  ASSERT_EQ(value, 5);
}

TEST(DataTest, OrderedSequencerIsSetUpWell) {
  using namespace torch::data::detail::sequencers; // NOLINT
  struct S {
    size_t sequence_number;
  };
  const size_t kMaxJobs = 5;
  OrderedSequencer<S> sequencer(kMaxJobs);
  ASSERT_EQ(sequencer.next_sequence_number_, 0);
  ASSERT_EQ(sequencer.buffer_.size(), kMaxJobs);
}

TEST(DataTest, OrderedSequencerReOrdersValues) {
  using namespace torch::data::detail::sequencers; // NOLINT
  struct S {
    size_t sequence_number;
  };
  const size_t kMaxJobs = 5;
  OrderedSequencer<S> sequencer(kMaxJobs);

  std::vector<size_t> v = {0, 2, 4, 3, 1};
  size_t index = 0;
  auto getter = [&v, &index]() { return S{v.at(index++)}; };

  // Let's say the sequence number matches for the first one, then it should
  // return immediately.
  const auto first = sequencer.next(getter);
  ASSERT_EQ(first.value().sequence_number, 0);
  ASSERT_EQ(index, 1);

  // Now it should call the getter until it gets the next value.
  ASSERT_EQ(1, sequencer.next(getter).value().sequence_number);
  ASSERT_EQ(index, 5);

  // The next three should come in order.
  for (size_t i = 2; i <= 4; ++i) {
    // New value doesn't matter. In fact, it shouldn't be accessed.
    ASSERT_EQ(i, sequencer.next(getter).value().sequence_number);
    // The index doesn't change.
    ASSERT_EQ(index, 5);
  }
}

TEST(DataTest, BatchLambdaAppliesFunctionToBatch) {
  using InputBatch = std::vector<int>;
  using OutputBatch = std::string;
  DummyDataset d;
  auto e = d.map(transforms::BatchLambda<InputBatch, OutputBatch>(
      [](std::vector<int> input) {
        return std::to_string(std::accumulate(input.begin(), input.end(), 0));
      }));
  ASSERT_EQ(e.get_batch({1, 2, 3, 4, 5}), std::string("20"));
}

TEST(DataTest, LambdaAppliesFunctionToExample) {
  auto d = DummyDataset().map(transforms::Lambda<int, std::string>(
      static_cast<std::string (*)(int)>(std::to_string)));
  std::vector<std::string> expected = {"1", "2", "3", "4", "5"};
  ASSERT_EQ(d.get_batch({0, 1, 2, 3, 4}), expected);
}

TEST(DataTest, CollateReducesBatch) {
  auto d =
      DummyDataset().map(transforms::Collate<int>([](std::vector<int> input) {
        return std::accumulate(input.begin(), input.end(), 0);
      }));
  ASSERT_EQ(d.get_batch({1, 2, 3, 4, 5}), 20);
}

TEST(DataTest, CollationReducesBatch) {
  struct Summer : transforms::Collation<int> {
    int apply_batch(std::vector<int> input) override {
      return std::accumulate(input.begin(), input.end(), 0);
    }
  };
  auto d = DummyDataset().map(Summer{});
  ASSERT_EQ(d.get_batch({1, 2, 3, 4, 5}), 20);
}

TEST(DataTest, SequentialSamplerReturnsIndicesInOrder) {
  samplers::SequentialSampler sampler(10);
  ASSERT_EQ(sampler.next(3).value(), std::vector<size_t>({0, 1, 2}));
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({3, 4, 5, 6, 7}));
  ASSERT_EQ(sampler.next(2).value(), std::vector<size_t>({8, 9}));
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SequentialSamplerReturnsLessValuesForLastBatch) {
  samplers::SequentialSampler sampler(5);
  ASSERT_EQ(sampler.next(3).value(), std::vector<size_t>({0, 1, 2}));
  ASSERT_EQ(sampler.next(100).value(), std::vector<size_t>({3, 4}));
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SequentialSamplerResetsWell) {
  samplers::SequentialSampler sampler(5);
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({0, 1, 2, 3, 4}));
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset();
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({0, 1, 2, 3, 4}));
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, RandomSamplerReturnsIndicesInCorrectRange) {
  samplers::RandomSampler sampler(10);

  std::vector<size_t> indices = sampler.next(3).value();
  for (auto i : indices) {
    ASSERT_GE(i, 0);
    ASSERT_LT(i, 10);
  }

  indices = sampler.next(5).value();
  for (auto i : indices) {
    ASSERT_GE(i, 0);
    ASSERT_LT(i, 10);
  }

  indices = sampler.next(2).value();
  for (auto i : indices) {
    ASSERT_GE(i, 0);
    ASSERT_LT(i, 10);
  }

  ASSERT_FALSE(sampler.next(10).has_value());
}

TEST(DataTest, RandomSamplerReturnsLessValuesForLastBatch) {
  samplers::RandomSampler sampler(5);
  ASSERT_EQ(sampler.next(3).value().size(), 3);
  ASSERT_EQ(sampler.next(100).value().size(), 2);
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, RandomSamplerResetsWell) {
  samplers::RandomSampler sampler(5);
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset();
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, TensorDatasetConstructsFromSingleTensor) {
  datasets::TensorDataset dataset(torch::eye(5));
  ASSERT_TRUE(
      torch::tensor({0, 0, 1, 0, 0}, torch::kFloat32).allclose(dataset.get(2)));
}

TEST(DataTest, TensorDatasetConstructsFromInitializerListOfTensors) {
  std::vector<torch::Tensor> vector = torch::eye(5).chunk(5);
  datasets::TensorDataset dataset(vector);
  ASSERT_TRUE(
      torch::tensor({0, 0, 1, 0, 0}, torch::kFloat32).allclose(dataset.get(2)));
}

TEST(DataTest, StackTransformWorksForExample) {
  struct D : public datasets::Dataset<D> {
    Example<> get(size_t index) override {
      return {tensor[index], 1 + tensor[index]};
    }

    size_t size() const override {
      return tensor.size(0);
    }

    torch::Tensor tensor{torch::eye(4)};
  };

  auto d = D().map(transforms::Stack<Example<>>());

  Example<> first = d.get_batch({0, 1});
  ASSERT_TRUE(first.data.allclose(torch::eye(4).slice(/*dim=*/0, 0, 2)));
  ASSERT_TRUE(first.target.allclose(1 + torch::eye(4).slice(/*dim=*/0, 0, 2)));

  Example<> second = d.get_batch({2, 3});
  ASSERT_TRUE(second.data.allclose(torch::eye(4).slice(/*dim=*/0, 2, 4)));
  ASSERT_TRUE(second.target.allclose(1 + torch::eye(4).slice(/*dim=*/0, 2, 4)));
}

TEST(DataTest, StackTransformWorksForTensorExample) {
  auto d = datasets::TensorDataset(torch::eye(4))
               .map(transforms::Stack<TensorExample>());

  TensorExample first = d.get_batch({0, 1});
  ASSERT_TRUE(first.data.allclose(torch::eye(4).slice(/*dim=*/0, 0, 2)));

  TensorExample second = d.get_batch({2, 3});
  ASSERT_TRUE(second.data.allclose(torch::eye(4).slice(/*dim=*/0, 2, 4)));
}

// Template classes cannot be nested in functions.
template <typename Target>
struct T : transforms::TensorTransform<Target> {
  torch::Tensor operator()(torch::Tensor input) override {
    return input * 2;
  }
};

struct TensorStringDataset
    : datasets::
          Dataset<TensorStringDataset, Example<torch::Tensor, std::string>> {
  Example<torch::Tensor, std::string> get(size_t index) override {
    return {torch::tensor(static_cast<double>(index)), std::to_string(index)};
  }

  size_t size() const override {
    return 100;
  }
};

TEST(DataTest, TensorTransformWorksForAnyTargetType) {
  auto d = TensorStringDataset().map(T<std::string>{});
  std::vector<Example<torch::Tensor, std::string>> batch = d.get_batch({1, 2});

  ASSERT_EQ(batch.size(), 2);
  ASSERT_TRUE(batch[0].data.allclose(torch::tensor(2.0)));
  ASSERT_EQ(batch[0].target, "1");

  ASSERT_TRUE(batch[1].data.allclose(torch::tensor(4.0)));
  ASSERT_EQ(batch[1].target, "2");
}

TEST(DataTest, TensorLambdaWorksforAnyTargetType) {
  auto d = TensorStringDataset().map(transforms::TensorLambda<std::string>(
      [](torch::Tensor input) { return input * 2; }));
  std::vector<Example<torch::Tensor, std::string>> batch = d.get_batch({1, 2});

  ASSERT_EQ(batch.size(), 2);
  ASSERT_TRUE(batch[0].data.allclose(torch::tensor(2.0)));
  ASSERT_EQ(batch[0].target, "1");

  ASSERT_TRUE(batch[1].data.allclose(torch::tensor(4.0)));
  ASSERT_EQ(batch[1].target, "2");
}

TEST(DataTest, QueuePushAndPopFromSameThread) {
  torch::data::detail::Queue<int> queue;
  queue.push(1);
  queue.push(2);
  ASSERT_EQ(queue.pop(), 1);
  ASSERT_EQ(queue.pop(), 2);
}

TEST(DataTest, QueuePopWithTimeoutThrowsUponTimeout) {
  torch::data::detail::Queue<int> queue;
  ASSERT_THROWS_WITH(
      queue.pop(10 * kMillisecond),
      "Timeout in DataLoader queue while waiting for next batch "
      "(timeout was 10 ms)");
}

TEST(DataTest, QueuePushAndPopFromDifferentThreads) {
  using torch::data::detail::Queue;

  // First test: push first and the pop in thread.
  {
    Queue<int> queue;
    queue.push(1);
    auto future =
        std::async(std::launch::async, [&queue] { return queue.pop(); });
    ASSERT_EQ(future.get(), 1);
  }

  // Second test: attempt to pop first (and block), then push.
  {
    Queue<int> queue;
    std::thread thread([&queue] {
      std::this_thread::sleep_for(20 * kMillisecond);
      queue.push(123);
    });
    ASSERT_EQ(queue.pop(), 123);
    thread.join();
  }
}

TEST(DataTest, QueueClearEmptiesTheQueue) {
  torch::data::detail::Queue<int> queue;
  queue.push(1);
  queue.push(2);
  queue.push(3);
  ASSERT_EQ(queue.clear(), 3);
  ASSERT_THROWS_WITH(queue.pop(1 * kMillisecond), "Timeout");
}

TEST(DataTest, DataShuttleCanPushAndPopJob) {
  torch::data::detail::DataShuttle<int, int> shuttle;
  shuttle.push_job(1);
  shuttle.push_job(2);
  ASSERT_EQ(shuttle.pop_job(), 1);
  ASSERT_EQ(shuttle.pop_job(), 2);
}

TEST(DataTest, DataShuttleCanPushAndPopResult) {
  torch::data::detail::DataShuttle<int, int> shuttle;
  // pop_result() will only attempt to pop if there was a push_job() first.
  shuttle.push_job(1);
  shuttle.push_job(2);

  shuttle.pop_job();
  shuttle.push_result(1);
  ASSERT_EQ(shuttle.pop_result().value(), 1);

  shuttle.pop_job();
  shuttle.push_result(2);
  ASSERT_EQ(shuttle.pop_result().value(), 2);
}

TEST(DataTest, DataShuttlePopResultReturnsNulloptWhenNoJobsInFlight) {
  torch::data::detail::DataShuttle<int, int> shuttle;
  ASSERT_FALSE(shuttle.pop_result().has_value());
  shuttle.push_job(1);
  shuttle.pop_job();
  shuttle.push_result(1);
  ASSERT_EQ(shuttle.pop_result().value(), 1);
  ASSERT_FALSE(shuttle.pop_result().has_value());
  ASSERT_FALSE(shuttle.pop_result().has_value());
}

TEST(DataTest, DataShuttleDrainMeansPopResultReturnsNullopt) {
  torch::data::detail::DataShuttle<int, int> shuttle;
  shuttle.push_job(1);
  shuttle.push_result(1);
  shuttle.drain();
  ASSERT_FALSE(shuttle.pop_result().has_value());
}

TEST(DataTest, DataShuttlePopResultTimesOut) {
  torch::data::detail::DataShuttle<int, int> shuttle;
  shuttle.push_job(1);
  ASSERT_THROWS_WITH(shuttle.pop_result(10 * kMillisecond), "Timeout");
}

TEST(DataLoaderTest, DataLoaderOptionsDefaultAsExpected) {
  DataLoaderOptions partial_options;
  FullDataLoaderOptions full_options(partial_options);
  ASSERT_EQ(full_options.batch_size, 1);
  ASSERT_FALSE(full_options.drop_last);
  ASSERT_EQ(full_options.workers, 0);
  ASSERT_EQ(full_options.max_jobs, 0);
  ASSERT_FALSE(full_options.timeout.has_value());
  ASSERT_TRUE(full_options.enforce_ordering);
}

TEST(DataLoaderTest, DataLoaderOptionsCoalesceOptionalValues) {
  auto partial_options = DataLoaderOptions(32).workers(10);
  FullDataLoaderOptions full_options(partial_options);
  ASSERT_EQ(full_options.batch_size, 32);
  ASSERT_EQ(full_options.max_jobs, 2 * 10);
}

TEST(DataLoaderTest, IteratorsCompareEqualToThemselves) {
  auto data_loader = torch::data::make_data_loader(DummyDataset(), 32);
  auto begin = data_loader->begin();
  ASSERT_EQ(begin, begin);
  auto end = data_loader->end();
  ASSERT_EQ(end, end);
}

TEST(DataLoaderTest, ValidIteratorsCompareUnequalToEachOther) {
  auto data_loader = torch::data::make_data_loader(DummyDataset(), 32);
  auto i = data_loader->begin();
  auto j = data_loader->begin();
  ASSERT_NE(i, j);
  ++j;
  ASSERT_NE(i, j);
}

TEST(DataLoaderTest, SentinelIteratorsCompareEqualToEachOther) {
  auto data_loader = torch::data::make_data_loader(DummyDataset(), 32);
  auto i = data_loader->end();
  auto j = data_loader->end();
  ASSERT_EQ(i, j);
}

TEST(DataLoaderTest, IteratorsCompareEqualToSentinelWhenExhausted) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size() / 4);
  auto i = data_loader->begin();
  auto end = data_loader->end();
  ASSERT_NE(i, end);
  ++i;
  ASSERT_NE(i, end);
  ++i;
  ASSERT_NE(i, end);
  ++i;
  ASSERT_NE(i, end);
  ++i;
  ASSERT_EQ(i, end);
}

TEST(DataLoaderTest, IteratorsShareState) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size() / 2);
  auto i = data_loader->begin();
  auto j = i;
  auto end = data_loader->end();
  ASSERT_NE(i, end);
  ASSERT_NE(j, end);
  ++i;
  ASSERT_NE(i, end);
  ASSERT_NE(j, end);
  ++j;
  ASSERT_EQ(i, end);
  ASSERT_EQ(j, end);
}

TEST(DataLoaderTest, CanDereferenceIteratorMultipleTimes) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size());
  auto i = data_loader->begin();
  ASSERT_NE(i, data_loader->end());
  ASSERT_EQ(i->size(), dataset.size());
  ASSERT_NE(i, data_loader->end());
  ASSERT_EQ(i->size(), dataset.size());
  ASSERT_NE(i, data_loader->end());
  ASSERT_EQ(i->size(), dataset.size());
  ASSERT_EQ(++i, data_loader->end());
}

TEST(DataLoaderTest, CallingBeginWhileOtherIteratorIsInFlightThrows) {
  DummyDataset dataset;
  auto data_loader =
      torch::data::make_data_loader(dataset, DataLoaderOptions(1).workers(2));
  auto i = data_loader->begin();
  ASSERT_THROWS_WITH(
      data_loader->begin(),
      "Attempted to get a new DataLoader iterator "
      "while another iterator is not yet exhausted");
}

TEST(DataLoaderTest, IncrementingExhaustedValidIteratorThrows) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size());
  auto i = data_loader->begin();
  ASSERT_NO_THROW(++i);
  ASSERT_THROWS_WITH(++i, "Attempted to increment iterator past the end");
}

TEST(DataLoaderTest, DereferencingExhaustedValidIteratorThrows) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size());
  auto i = data_loader->begin();
  ASSERT_NO_THROW(++i);
  ASSERT_THROWS_WITH(
      *i, "Attempted to dereference iterator that was past the end");
}

TEST(DataLoaderTest, IncrementingSentinelIteratorThrows) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size());
  auto i = data_loader->end();
  ASSERT_THROWS_WITH(
      ++i,
      "Incrementing the DataLoader's past-the-end iterator is not allowed");
}

TEST(DataLoaderTest, DereferencingSentinelIteratorThrows) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size());
  auto i = data_loader->end();
  ASSERT_THROWS_WITH(
      *i,
      "Dereferencing the DataLoader's past-the-end iterator is not allowed");
}

TEST(DataLoaderTest, ThrowsWhenBatchSizeExceedsDatasetSize) {
  DummyDataset dataset;
  ASSERT_THROWS_WITH(
      torch::data::make_data_loader(dataset, dataset.size() + 1),
      "Batch size (was 101) must not be larger "
      "than the dataset size (was 100)");
}

TEST(DataLoaderTest, YieldsCorrectBatchSize) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, 25);
  auto iterator = data_loader->begin();
  ASSERT_EQ(iterator->size(), 25);
  ASSERT_EQ((++iterator)->size(), 25);
  ASSERT_EQ((++iterator)->size(), 25);
  ASSERT_EQ((++iterator)->size(), 25);
  ASSERT_EQ(++iterator, data_loader->end());
}

TEST(
    DataLoaderTest,
    ReturnsLastBatchWhenSmallerThanBatchSizeWhenDropLastIsFalse) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(33).drop_last(false));
  auto iterator = data_loader->begin();
  ASSERT_EQ(iterator->size(), 33);
  ASSERT_EQ((++iterator)->size(), 33);
  ASSERT_EQ((++iterator)->size(), 33);
  ASSERT_EQ((++iterator)->size(), 1);
  ASSERT_EQ(++iterator, data_loader->end());
}

TEST(
    DataLoaderTest,
    DoesNotReturnLastBatchWhenSmallerThanBatchSizeWhenDropLastIsTrue) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(33).drop_last(true));
  auto iterator = data_loader->begin();
  ASSERT_EQ(iterator->size(), 33);
  ASSERT_EQ((++iterator)->size(), 33);
  ASSERT_EQ((++iterator)->size(), 33);
  ASSERT_EQ(++iterator, data_loader->end());
}

TEST(DataLoaderTest, RespectsTimeout) {
  struct Baton {
    std::condition_variable cv;
    std::mutex mutex;
  };

  struct D : datasets::Dataset<DummyDataset, int> {
    D(std::shared_ptr<Baton> b) : baton(std::move(b)) {}
    int get(size_t index) override {
      std::unique_lock<std::mutex> lock(baton->mutex);
      baton->cv.wait_for(lock, 1000 * kMillisecond);
      return 0;
    }
    size_t size() const override {
      return 100;
    }
    std::shared_ptr<Baton> baton;
  };

  auto baton = std::make_shared<Baton>();

  auto data_loader = torch::data::make_data_loader(
      D{baton}, DataLoaderOptions().workers(1).timeout(10 * kMillisecond));

  auto start = std::chrono::system_clock::now();

  ASSERT_THROWS_WITH(*data_loader->begin(), "Timeout");
  baton->cv.notify_one();

  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  ASSERT_LT(duration.count(), 1);
}

struct OrderingTestDataset : datasets::BatchDataset<DummyDataset, int> {
  int get_batch(torch::ArrayRef<size_t> indices) override {
    static int thread_counter = 0;
    thread_local int thread_id = thread_counter++;
    static std::condition_variable cv;
    static std::mutex mutex;
    static std::array<size_t, 4> order = {3, 1, 0, 2};
    static std::atomic<size_t> index{0};

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return order.at(index) == thread_id; });
    ++index;
    cv.notify_all();
    lock.unlock();
    return thread_id;
  }

  size_t size() const override {
    return 4;
  }
};

TEST(DataLoaderTest, EnforcesOrderingAmongThreadsWhenConfigured) {
  auto data_loader = torch::data::make_data_loader(
      OrderingTestDataset{},
      DataLoaderOptions().batch_size(1).workers(4).enforce_ordering(true));
  size_t index = 0;
  for (int value : *data_loader) {
    ASSERT_EQ(value, index++);
  }
}

TEST(DataLoaderTest, Reset) {
  DummyDataset dataset;
  auto data_loader = torch::data::make_data_loader(dataset, dataset.size() / 2);
  auto end = data_loader->end();

  auto iterator = data_loader->begin();
  ASSERT_NE(iterator, end);
  ASSERT_NE(++iterator, end);
  ASSERT_EQ(++iterator, end);

  iterator = data_loader->begin();
  ASSERT_NE(iterator, end);
  ASSERT_NE(++iterator, end);
  ASSERT_EQ(++iterator, end);

  iterator = data_loader->begin();
  ASSERT_NE(iterator, end);
  ASSERT_NE(++iterator, end);
  ASSERT_EQ(++iterator, end);
}

TEST(DataLoaderTest, TestExceptionsArePropagatedFromWorkers) {
  struct D : datasets::Dataset<DummyDataset, int> {
    int get(size_t index) override {
      throw std::invalid_argument("badness");
    }
    size_t size() const override {
      return 100;
    }
  };

  auto data_loader =
      torch::data::make_data_loader(D{}, DataLoaderOptions().workers(2));
  auto iterator = data_loader->begin();

  try {
    (void)*iterator;
  } catch (torch::data::WorkerException& e) {
    ASSERT_EQ(
        e.what(),
        std::string("Caught exception in DataLoader worker thread. "
                    "Original message: badness"));
    ASSERT_THROW(
        std::rethrow_exception(e.original_exception), std::invalid_argument);
  }
}
