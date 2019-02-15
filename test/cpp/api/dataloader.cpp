#include <gtest/gtest.h>

#include <torch/data.h>
#include <torch/data/detail/sequencers.h>
#include <torch/serialize.h>
#include <torch/types.h>

#include <test/cpp/api/support.h>

#include <c10/util/ArrayRef.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

using namespace torch::data; // NOLINT

const std::chrono::milliseconds kMillisecond(1);

struct DummyDataset : datasets::Dataset<DummyDataset, int> {
  explicit DummyDataset(size_t size = 100) : size_(size) {}

  int get(size_t index) override {
    return 1 + index;
  }
  torch::optional<size_t> size() const override {
    return size_;
  }

  size_t size_;
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

// dummy chunk data reader with 3 chunks and 35 examples in total. Each chunk
// contains 10, 5, 20 examples respectively.
struct DummyChunkDataReader
    : public datasets::ChunkDataReader<std::vector<int>> {
 public:
  using BatchType = std::vector<int>;

  /// Read an entire chunk.
  BatchType read_chunk(size_t chunk_index) override {
    BatchType batch_data;
    int start_index = chunk_index == 0
        ? 0
        : std::accumulate(chunk_sizes, chunk_sizes + chunk_index, 0);

    batch_data.resize(chunk_sizes[chunk_index]);

    std::iota(batch_data.begin(), batch_data.end(), start_index);

    return batch_data;
  }

  size_t chunk_count() override {
    return chunk_count_;
  };

  void reset() override{};

  const static size_t chunk_count_ = 3;
  size_t chunk_sizes[chunk_count_] = {10, 5, 20};
};

TEST(DataTest, ChunkDataSetWithInvalidInitParameter) {
  DummyChunkDataReader data_reader;
  samplers::SequentialSampler sampler(0);

  auto initialization_function =
      [&](size_t preloader_count, size_t batch_size, size_t cache_size) {
        datasets::SharedBatchDataset<datasets::ChunkDataset<
            DummyChunkDataReader,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>
            dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
                DummyChunkDataReader,
                samplers::SequentialSampler,
                samplers::SequentialSampler>>(
                data_reader,
                sampler,
                sampler,
                datasets::ChunkDatasetOptions(
                    preloader_count, batch_size, cache_size));
      };

  ASSERT_THROWS_WITH(
      initialization_function(0, 1, 1),
      "Preloader count is 0. At least one preloader needs to be specified.");

  ASSERT_THROWS_WITH(
      initialization_function(1, 0, 1),
      "Batch size is 0. A positive batch size needs to be specified.");

  ASSERT_THROWS_WITH(
      initialization_function(1, 1, 0),
      "Cache size is 0. A positive cache size needs to be specified.");

  ASSERT_THROWS_WITH(
      initialization_function(1, 10, 5),
      "Cache size is less than batch size. Cache needs to be large enough to "
      "hold at least one batch.");
}

struct InfiniteStreamDataset
    : datasets::StreamDataset<InfiniteStreamDataset, std::vector<int>> {
  std::vector<int> get_batch(size_t batch_size) override {
    std::vector<int> batch(batch_size);
    for (auto& i : batch) {
      i = counter++;
    }
    return batch;
  }

  torch::optional<size_t> size() const override {
    return torch::nullopt;
  }

  size_t counter = 0;
};

TEST(DataTest, InfiniteStreamDataset) {
  const size_t kBatchSize = 13;

  auto dataset = InfiniteStreamDataset().map(
      transforms::Lambda<int>([](int x) { return x + 1; }));

  auto data_loader = torch::data::make_data_loader(
      std::move(dataset),
      samplers::StreamSampler(/*epoch_size=*/39),
      kBatchSize);

  size_t batch_index = 0;
  for (auto& batch : *data_loader) {
    ASSERT_LT(batch_index, 3);
    ASSERT_EQ(batch.size(), kBatchSize);
    for (size_t j = 0; j < kBatchSize; ++j) {
      ASSERT_EQ(batch.at(j), 1 + (batch_index * kBatchSize) + j);
    }
    batch_index += 1;
  }
  ASSERT_EQ(batch_index, 3);
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

  // Let's say the sequence number matches for the batch one, then it should
  // return immediately.
  const auto batch = sequencer.next(getter);
  ASSERT_EQ(batch.value().sequence_number, 0);
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

TEST(DataTest, SequentialSamplerResetsWithNewSizeWell) {
  samplers::SequentialSampler sampler(5);
  ASSERT_EQ(sampler.next(5).value(), std::vector<size_t>({0, 1, 2, 3, 4}));
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset(7);
  ASSERT_EQ(
      sampler.next(7).value(), std::vector<size_t>({0, 1, 2, 3, 4, 5, 6}));
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset(3);
  ASSERT_EQ(sampler.next(3).value(), std::vector<size_t>({0, 1, 2}));
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, CanSaveAndLoadSequentialSampler) {
  {
    samplers::SequentialSampler a(10);
    ASSERT_EQ(a.index(), 0);
    std::stringstream stream;
    torch::save(a, stream);

    samplers::SequentialSampler b(10);
    torch::load(b, stream);
    ASSERT_EQ(b.index(), 0);
  }
  {
    samplers::SequentialSampler a(10);
    a.next(3);
    a.next(4);
    ASSERT_EQ(a.index(), 7);
    std::stringstream stream;
    torch::save(a, stream);

    samplers::SequentialSampler b(10);
    torch::load(b, stream);
    ASSERT_EQ(b.index(), 7);
  }
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

TEST(DataTest, RandomSamplerResetsWithNewSizeWell) {
  samplers::RandomSampler sampler(5);
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset(7);
  ASSERT_EQ(sampler.next(7).value().size(), 7);
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset(3);
  ASSERT_EQ(sampler.next(3).value().size(), 3);
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, SavingAndLoadingRandomSamplerYieldsSameSequence) {
  {
    samplers::RandomSampler a(10);

    std::stringstream stream;
    torch::save(a, stream);

    samplers::RandomSampler b(10);
    torch::load(b, stream);

    ASSERT_EQ(a.next(10).value(), b.next(10).value());
  }
  {
    samplers::RandomSampler a(10);
    a.next(3);
    ASSERT_EQ(a.index(), 3);

    std::stringstream stream;
    torch::save(a, stream);

    samplers::RandomSampler b(10);
    torch::load(b, stream);
    ASSERT_EQ(b.index(), 3);

    auto b_sequence = b.next(10).value();
    ASSERT_EQ(b_sequence.size(), 7);
    ASSERT_EQ(a.next(10).value(), b_sequence);
  }
}

TEST(DataTest, StreamSamplerReturnsTheBatchSizeAndThenRemainder) {
  samplers::StreamSampler sampler(/*epoch_size=*/100);
  ASSERT_EQ(sampler.next(10).value(), 10);
  ASSERT_EQ(sampler.next(2).value(), 2);
  ASSERT_EQ(sampler.next(85).value(), 85);
  ASSERT_EQ(sampler.next(123).value(), 3);
  ASSERT_FALSE(sampler.next(1).has_value());
}

TEST(DataTest, StreamSamplerResetsWell) {
  samplers::StreamSampler sampler(/*epoch_size=*/5);
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset();
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  ASSERT_FALSE(sampler.next(2).has_value());
}

TEST(DataTest, StreamSamplerResetsWithNewSizeWell) {
  samplers::StreamSampler sampler(/*epoch_size=*/5);
  ASSERT_EQ(sampler.next(5).value().size(), 5);
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset(7);
  ASSERT_EQ(sampler.next(7).value().size(), 7);
  ASSERT_FALSE(sampler.next(2).has_value());
  sampler.reset(3);
  ASSERT_EQ(sampler.next(3).value().size(), 3);
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

    torch::optional<size_t> size() const override {
      return tensor.size(0);
    }

    torch::Tensor tensor{torch::eye(4)};
  };

  auto d = D().map(transforms::Stack<Example<>>());

  Example<> batch = d.get_batch({0, 1});
  ASSERT_TRUE(batch.data.allclose(torch::eye(4).slice(/*dim=*/0, 0, 2)));
  ASSERT_TRUE(batch.target.allclose(1 + torch::eye(4).slice(/*dim=*/0, 0, 2)));

  Example<> second = d.get_batch({2, 3});
  ASSERT_TRUE(second.data.allclose(torch::eye(4).slice(/*dim=*/0, 2, 4)));
  ASSERT_TRUE(second.target.allclose(1 + torch::eye(4).slice(/*dim=*/0, 2, 4)));
}

TEST(DataTest, StackTransformWorksForTensorExample) {
  auto d = datasets::TensorDataset(torch::eye(4))
               .map(transforms::Stack<TensorExample>());

  TensorExample batch = d.get_batch({0, 1});
  ASSERT_TRUE(batch.data.allclose(torch::eye(4).slice(/*dim=*/0, 0, 2)));

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

  torch::optional<size_t> size() const override {
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

struct DummyTensorDataset
    : datasets::Dataset<DummyTensorDataset, Example<torch::Tensor, int>> {
  Example<torch::Tensor, int> get(size_t index) override {
    const auto channels = static_cast<int64_t>(index);
    torch::Tensor tensor =
        (channels > 0) ? torch::ones({channels, 4, 4}) : torch::ones({4, 4});
    return {tensor, static_cast<int>(channels)};
  }

  torch::optional<size_t> size() const override {
    return 100;
  }
};

TEST(DataTest, NormalizeTransform) {
  auto dataset = DummyTensorDataset().map(transforms::Normalize<int>(0.5, 0.1));

  // Works for zero (one implicit) channels
  std::vector<Example<torch::Tensor, int>> output = dataset.get_batch(0);
  ASSERT_EQ(output.size(), 1);
  // (1 - 0.5) / 0.1 = 5
  ASSERT_TRUE(output[0].data.allclose(torch::ones({4, 4}) * 5))
      << output[0].data;

  // Works for one explicit channel
  output = dataset.get_batch(1);
  ASSERT_EQ(output.size(), 1);
  ASSERT_EQ(output[0].data.size(0), 1);
  ASSERT_TRUE(output[0].data.allclose(torch::ones({1, 4, 4}) * 5))
      << output[0].data;

  // Works for two channels with different moments
  dataset = DummyTensorDataset().map(
      transforms::Normalize<int>({0.5, 1.5}, {0.1, 0.2}));
  output = dataset.get_batch(2);
  ASSERT_EQ(output.size(), 1);
  ASSERT_EQ(output[0].data.size(0), 2);
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
                  .allclose(torch::ones({1, 4, 4}) * 5))
      << output[0].data;
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/1)
                  .allclose(torch::ones({1, 4, 4}) * -2.5))
      << output[0].data;

  // Works for three channels with one moment value
  dataset = DummyTensorDataset().map(transforms::Normalize<int>(1.5, 0.2));
  output = dataset.get_batch(3);
  ASSERT_EQ(output.size(), 1);
  ASSERT_EQ(output[0].data.size(0), 3);
  ASSERT_TRUE(output[0].data.allclose(torch::ones({3, 4, 4}) * -2.5))
      << output[0].data;

  // Works for three channels with different moments
  dataset = DummyTensorDataset().map(
      transforms::Normalize<int>({0.5, 1.5, -1.5}, {0.1, 0.2, 0.2}));
  output = dataset.get_batch(3);
  ASSERT_EQ(output.size(), 1);
  ASSERT_EQ(output[0].data.size(0), 3);
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
                  .allclose(torch::ones({1, 4, 4}) * 5))
      << output[0].data;
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/1, /*end=*/2)
                  .allclose(torch::ones({1, 4, 4}) * -2.5))
      << output[0].data;
  ASSERT_TRUE(output[0]
                  .data.slice(/*dim=*/0, /*start=*/2)
                  .allclose(torch::ones({1, 4, 4}) * 12.5))
      << output[0].data;
}

struct UnCopyableDataset : public datasets::Dataset<UnCopyableDataset> {
  UnCopyableDataset() = default;

  UnCopyableDataset(const UnCopyableDataset&) = delete;
  UnCopyableDataset& operator=(const UnCopyableDataset&) = delete;

  UnCopyableDataset(UnCopyableDataset&&) = default;
  UnCopyableDataset& operator=(UnCopyableDataset&&) = default;

  ~UnCopyableDataset() = default;

  Example<> get(size_t index) override {
    return {torch::tensor(static_cast<int64_t>(index)),
            torch::tensor(static_cast<int64_t>(index))};
  }

  torch::optional<size_t> size() const override {
    return 100;
  }
};

TEST(DataTest, MapDoesNotCopy) {
  auto dataset = UnCopyableDataset()
                     .map(transforms::TensorLambda<>(
                         [](torch::Tensor tensor) { return tensor + 1; }))
                     .map(transforms::TensorLambda<>(
                         [](torch::Tensor tensor) { return tensor + 2; }))
                     .map(transforms::TensorLambda<>(
                         [](torch::Tensor tensor) { return tensor + 3; }));

  auto data = dataset.get_batch(1).at(0).data;
  ASSERT_EQ(data.numel(), 1);
  ASSERT_EQ(data[0].item<float>(), 7);
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

  // First test: push batch and the pop in thread.
  {
    Queue<int> queue;
    queue.push(1);
    auto future =
        std::async(std::launch::async, [&queue] { return queue.pop(); });
    ASSERT_EQ(future.get(), 1);
  }

  // Second test: attempt to pop batch (and block), then push.
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
  // pop_result() will only attempt to pop if there was a push_job() batch.
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

struct UncopyableDataset : datasets::Dataset<UncopyableDataset, int> {
  UncopyableDataset(const std::string& /* unused */) {}

  UncopyableDataset(UncopyableDataset&&) = default;
  UncopyableDataset& operator=(UncopyableDataset&&) = default;

  UncopyableDataset(const UncopyableDataset&) = delete;
  UncopyableDataset& operator=(const UncopyableDataset&) = delete;

  int get(size_t index) override {
    return 1 + index;
  }
  torch::optional<size_t> size() const override {
    return 100;
  }
};

TEST(DataTest, SharedBatchDatasetReallyIsShared) {
  // This test will only compile if we really are not making any copies.
  // There is otherwise no logic to test and because it is not deterministic
  // how many and when worker threads access the shareddataset, we don't have
  // any additional assertions here.

  auto shared_dataset =
      torch::data::datasets::make_shared_dataset<UncopyableDataset>(
          "uncopyable");

  auto data_loader = torch::data::make_data_loader(
      shared_dataset, torch::data::DataLoaderOptions().workers(3));

  for (auto batch : *data_loader) {
    /* exhaust */
  }
}

TEST(DataTest, SharedBatchDatasetDoesNotIncurCopyWhenPassedDatasetObject) {
  // This will not compile if a copy is made.
  auto shared_dataset =
      torch::data::datasets::make_shared_dataset<UncopyableDataset>(
          UncopyableDataset("uncopyable"));
  ASSERT_EQ(shared_dataset.size().value(), 100);
}

struct TestIndex : public torch::data::samplers::CustomBatchRequest {
  explicit TestIndex(size_t offset, std::vector<size_t> index)
      : offset(offset), index(std::move(index)) {}
  size_t size() const override {
    return index.size();
  }
  size_t offset;
  std::vector<size_t> index;
};

struct TestIndexDataset
    : datasets::BatchDataset<TestIndexDataset, std::vector<int>, TestIndex> {
  explicit TestIndexDataset(size_t size) : data(size) {
    std::iota(data.begin(), data.end(), size_t(0));
  }
  std::vector<int> get_batch(TestIndex index) override {
    std::vector<int> batch;
    for (auto i : index.index) {
      batch.push_back(index.offset + data.at(i));
    }
    return batch;
  }
  torch::optional<size_t> size() const override {
    return data.size();
  }
  std::vector<int> data;
};

struct TestIndexSampler : public samplers::Sampler<TestIndex> {
  explicit TestIndexSampler(size_t size) : size_(size) {}
  void reset(torch::optional<size_t> new_size = torch::nullopt) override {}
  torch::optional<TestIndex> next(size_t batch_size) override {
    if (index_ >= size_) {
      return torch::nullopt;
    }
    std::vector<size_t> indices(batch_size);
    std::iota(indices.begin(), indices.end(), size_t(0));
    index_ += batch_size;
    return TestIndex(batch_size, std::move(indices));
  }
  void save(torch::serialize::OutputArchive& archive) const override {}
  void load(torch::serialize::InputArchive& archive) override {}
  size_t index_ = 0;
  size_t size_;
};

TEST(DataTest, CanUseCustomTypeAsIndexType) {
  const int kBatchSize = 10;
  auto data_loader = torch::data::make_data_loader(
      TestIndexDataset(23), TestIndexSampler(23), kBatchSize);

  size_t i = 0;
  for (auto batch : *data_loader) {
    for (int j = 0; j < kBatchSize; ++j) {
      ASSERT_EQ(batch.at(j), 10 + j);
    }
    i += 1;
  }
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

TEST(DataLoaderTest, MakeDataLoaderDefaultsAsExpected) {
  auto data_loader = torch::data::make_data_loader(
      DummyDataset().map(transforms::Lambda<int>([](int x) { return x + 1; })));
  ASSERT_EQ(data_loader->options().batch_size, 1);
}

struct UnsizedDataset : public datasets::Dataset<UnsizedDataset> {
  torch::data::Example<> get(size_t i) {
    return {torch::ones(i), torch::ones(i)};
  }
  torch::optional<size_t> size() const noexcept {
    return torch::nullopt;
  }
};

TEST(
    DataLoaderTest,
    MakeDataLoaderThrowsWhenConstructingSamplerWithUnsizedDataset) {
  ASSERT_THROWS_WITH(
      torch::data::make_data_loader(UnsizedDataset{}),
      "Expected the dataset to be sized in order to construct the Sampler");
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
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value() / 4);
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
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value() / 2);
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
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          dataset,
          /*batch_size=*/1);
  auto iterator = data_loader->begin();
  std::vector<int> expected = {1};
  ASSERT_EQ(*iterator, expected);
  ASSERT_EQ(*iterator, expected);
  ++iterator;
  expected[0] = 2;
  ASSERT_EQ(*iterator, expected);
  ASSERT_EQ(*iterator, expected);
  ++iterator;
  expected[0] = 3;
  ASSERT_EQ(*iterator, expected);
  ASSERT_EQ(*iterator, expected);
}

TEST(DataLoaderTest, CanUseIteratorAlgorithms) {
  struct D : datasets::BatchDataset<D, int> {
    int get_batch(torch::ArrayRef<size_t> indices) override {
      return 1 + indices.front();
    }
    torch::optional<size_t> size() const override {
      return 10;
    }
  };

  D dataset;
  auto data_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          dataset, 1);
  std::vector<int> values;
  std::copy(
      data_loader->begin(), data_loader->end(), std::back_inserter(values));
  std::vector<int> expected(dataset.size().value());
  std::iota(expected.begin(), expected.end(), size_t(1));
  ASSERT_EQ(values, expected);
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
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  auto i = data_loader->begin();
  ASSERT_NO_THROW(++i);
  ASSERT_THROWS_WITH(++i, "Attempted to increment iterator past the end");
}

TEST(DataLoaderTest, DereferencingExhaustedValidIteratorThrows) {
  DummyDataset dataset;
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  auto i = data_loader->begin();
  ASSERT_NO_THROW(++i);
  ASSERT_THROWS_WITH(
      *i, "Attempted to dereference iterator that was past the end");
}

TEST(DataLoaderTest, IncrementingSentinelIteratorThrows) {
  DummyDataset dataset;
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  auto i = data_loader->end();
  ASSERT_THROWS_WITH(
      ++i,
      "Incrementing the DataLoader's past-the-end iterator is not allowed");
}

TEST(DataLoaderTest, DereferencingSentinelIteratorThrows) {
  DummyDataset dataset;
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value());
  auto i = data_loader->end();
  ASSERT_THROWS_WITH(
      *i,
      "Dereferencing the DataLoader's past-the-end iterator is not allowed");
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
    torch::optional<size_t> size() const override {
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

// stackoverflow.com/questions/24465533/implementing-boostbarrier-in-c11
struct Barrier {
  explicit Barrier(size_t target) : counter_(target) {}
  void wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (--counter_ == 0) {
      cv_.notify_all();
    } else {
      cv_.wait(lock, [this] { return this->counter_ == 0; });
    }
  }

  size_t counter_;
  std::condition_variable cv_;
  std::mutex mutex_;
};

// On the OrderingTest: This test is intended to verify that the
// `enforce_ordering` option of the dataloader works correctly. The reason this
// flag exists is because when the dataloader has multiple workers (threads)
// enabled and this flag is not set, the order in which worker threads finish
// loading their respective batch and push it back to the dataloader's main
// thread (for outside consumption) is not deterministic. Imagine the sampler is
// a SequentialSampler with indices 0, 1, 2, 3. With batch size 1, each index
// will be a single "job". Inside the dataloader, worker threads block until a
// job is available. It is not deterministic which worker thread wakes up batch
// to dequeue a particular batch. Further, some worker threads may take longer
// than others to read the data for their index. As such, it could be that
// worker thread 2 finishes before all other threads and returns its batch to
// the main thread. In that case, the dataloader iterator would return the datum
// at index 2 batch, and afterwards the datum from whatever thread finishes
// next. As such, the user may see data from indices 2, 0, 3, 1. On another run
// of the same dataloader on the same data, threads may be scheduled differently
// and return in order 0, 2, 3, 1. To force this ordering to deterministically
// be 0, 1, 2, 3, the `enforce_ordering` flag can be set to true. In that case,
// the dataloader will use a *sequencer* internally which keeps track of which
// datum is expected next, and buffers any other results until that next
// expected value arrives. For example, workers 1, 2, 3 may finish before worker
// 0. If `enforce_ordering` is true, the sequencer will internally buffer the
// results from 1, 2, 3 until worker 0 finishes. Only then does the dataloader
// return the datum from worker 0 to the user (and then datum 1 the next time,
// then 2 and so on).
//
// The way the test works is that we start
// `kNumberOfWorkers` workers in the dataloader, which each get an index from a
// `SequentialSampler` in the range `0...kNumberOfWorkers-1`. Each worker thread
// has a copy of the dataset, and thus `get_batch()` is called on the
// thread-local copy in each worker. We want to simulate out-of-order completion
// of these threads. For this, we batch set a barrier in the `get_batch()`
// method to make sure every worker has some index to fetch assigned. Further,
// each worker thread has a unique ID in `0...kNumberOfWorkers-1`.
// There is a hard-coded ordering, `kOrderInWhichWorkersReturnTheirBatch`, in
// which we want the worker threads to return. For this, an iterator into this
// order is maintained. When the derferenced iterator (the current order index)
// matches the thread ID of a worker, it knows it can now return its index as
// well as progress the iterator. Inside the dataloader, the sequencer should
// buffer these indices such that they are ultimately returned in order.

namespace ordering_test {
namespace {
const size_t kNumberOfWorkers = 10;
const std::vector<size_t> kOrderInWhichWorkersReturnTheirBatch =
    {3, 7, 0, 5, 4, 8, 2, 1, 9, 6};
} // namespace

struct Dataset : datasets::BatchDataset<Dataset, size_t> {
  Dataset() = default;

  // This copy constructor will be called when we copy the dataset into a
  // particular thread.
  Dataset(const Dataset& other) {
    static std::atomic<size_t> counter{0};
    thread_id_ = counter.fetch_add(1);
  }

  Dataset(Dataset&& other) noexcept = default;
  Dataset& operator=(const Dataset& other) = delete;
  Dataset& operator=(Dataset&& other) noexcept = delete;

  size_t get_batch(torch::ArrayRef<size_t> indices) override {
    static Barrier barrier(kNumberOfWorkers);
    static auto order_iterator = kOrderInWhichWorkersReturnTheirBatch.begin();
    static std::condition_variable cv;
    static std::mutex mutex;

    // Wait for all threads to get an index batch and arrive here.
    barrier.wait();

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this] { return *order_iterator == this->thread_id_; });
    ++order_iterator;
    lock.unlock();
    cv.notify_all();

    return indices.front();
  }

  torch::optional<size_t> size() const override {
    return kNumberOfWorkers;
  }

  size_t thread_id_ = 0;
};

} // namespace ordering_test

TEST(DataLoaderTest, EnforcesOrderingAmongThreadsWhenConfigured) {
  auto data_loader = torch::data::make_data_loader(
      ordering_test::Dataset{},
      torch::data::samplers::SequentialSampler(ordering_test::kNumberOfWorkers),
      DataLoaderOptions()
          .batch_size(1)
          .workers(ordering_test::kNumberOfWorkers)
          .enforce_ordering(true));
  std::vector<size_t> output;
  for (size_t value : *data_loader) {
    output.push_back(value);
  }
  std::vector<size_t> expected(ordering_test::kNumberOfWorkers);
  std::iota(expected.begin(), expected.end(), size_t(0));
  ASSERT_EQ(expected, output);
}

TEST(DataLoaderTest, Reset) {
  DummyDataset dataset;
  auto data_loader =
      torch::data::make_data_loader(dataset, dataset.size().value() / 2);
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
    torch::optional<size_t> size() const override {
      return 100;
    }
  };

  auto data_loader = torch::data::make_data_loader(
      D{}, samplers::RandomSampler(100), DataLoaderOptions().workers(2));
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

TEST(DataLoaderTest, StatefulDatasetWithNoWorkers) {
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;

  struct D : datasets::StatefulDataset<D, int, size_t> {
    torch::optional<int> get_batch(size_t) override {
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        return counter++;
      }
      return torch::nullopt;
    }
    torch::optional<size_t> size() const override {
      return 100;
    }
    void reset() override {
      counter = 0;
    }
    int counter = 0;
  };

  auto data_loader = torch::data::make_data_loader(D{});

  for (size_t i = 0; i < 10; ++i) {
    const auto number_of_iterations =
        std::distance(data_loader->begin(), data_loader->end());
    ASSERT_EQ(
        number_of_iterations, kNumberOfExamplesAfterWhichTheDatasetExhausts)
        << "epoch " << i;
  }

  for (const int i : *data_loader) {
    ASSERT_LT(i, kNumberOfExamplesAfterWhichTheDatasetExhausts);
  }
}

TEST(DataLoaderTest, StatefulDatasetWithManyWorkers) {
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;
  const int kNumberOfWorkers = 4;

  struct D : datasets::StatefulDataset<D, int, size_t> {
    torch::optional<int> get_batch(size_t) override {
      std::lock_guard<std::mutex> lock(mutex);
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        return counter++;
      }
      return torch::nullopt;
    }
    torch::optional<size_t> size() const override {
      return 100;
    }
    void reset() override {
      counter = 0;
    }
    int counter = 0;
    std::mutex mutex;
  };

  auto data_loader = torch::data::make_data_loader(
      torch::data::datasets::make_shared_dataset<D>(),
      DataLoaderOptions().workers(kNumberOfWorkers));

  for (size_t i = 0; i < 10; ++i) {
    const auto number_of_iterations =
        std::distance(data_loader->begin(), data_loader->end());
    ASSERT_EQ(
        number_of_iterations, kNumberOfExamplesAfterWhichTheDatasetExhausts)
        << "epoch " << i;
  }

  for (const int i : *data_loader) {
    ASSERT_LT(i, kNumberOfExamplesAfterWhichTheDatasetExhausts);
  }
}

TEST(DataLoaderTest, StatefulDatasetWithMap) {
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;

  struct D : datasets::StatefulDataset<D, int, size_t> {
    torch::optional<int> get_batch(size_t) override {
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        return counter++;
      }
      return torch::nullopt;
    }
    torch::optional<size_t> size() const override {
      return 100;
    }
    void reset() override {
      counter = 0;
    }
    int counter = 0;
  };

  auto data_loader = torch::data::make_data_loader(
      D().map(transforms::BatchLambda<int, std::string>(
                  [](int x) { return std::to_string(x); }))
          .map(transforms::BatchLambda<std::string, torch::Tensor>(
              [](const std::string& x) {
                return torch::tensor(static_cast<int64_t>(std::stoi(x)));
              })),
      DataLoaderOptions{});

  for (size_t i = 0; i < 10; ++i) {
    const auto number_of_iterations =
        std::distance(data_loader->begin(), data_loader->end());
    ASSERT_EQ(
        number_of_iterations, kNumberOfExamplesAfterWhichTheDatasetExhausts)
        << "epoch " << i;
  }

  for (const torch::Tensor& t : *data_loader) {
    ASSERT_LT(t.item<int64_t>(), kNumberOfExamplesAfterWhichTheDatasetExhausts);
  }
}

TEST(DataLoaderTest, StatefulDatasetWithCollate) {
  const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;

  struct D : datasets::StatefulDataset<D> {
    torch::optional<std::vector<Example<>>> get_batch(
        size_t batch_size) override {
      if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        counter += batch_size;
        std::vector<Example<>> batch(
            /*count=*/batch_size,
            Example<>{torch::ones(batch_size + 1),
                      torch::zeros(batch_size - 1)});
        return batch;
      }
      return torch::nullopt;
    }
    torch::optional<size_t> size() const override {
      return 100;
    }
    void reset() override {
      counter = 0;
    }
    int counter = 0;
  };

  auto d = D().map(transforms::Stack<Example<>>());

  const size_t kBatchSize = 5;

  // Notice that the `get_batch()` of the dataset returns a vector<Example>, but
  // the `Stack` collation stacks the tensors into one.
  torch::optional<Example<>> batch = d.get_batch(kBatchSize);
  ASSERT_TRUE(batch.has_value());
  ASSERT_EQ(batch->data.size(0), kBatchSize);
  ASSERT_EQ(batch->data.size(1), kBatchSize + 1);
  ASSERT_EQ(batch->target.size(0), kBatchSize);
  ASSERT_EQ(batch->target.size(1), kBatchSize - 1);

  ASSERT_TRUE(batch->data[0].allclose(torch::ones(kBatchSize + 1)));
  ASSERT_TRUE(batch->target[0].allclose(torch::zeros(kBatchSize - 1)));
}

// This test tests the core function for iterate through a chunk dataset. It
// contains test cases with different parameter combination. (For example,
// different prefetch count, batch size and data loader worker count). It
// verifies the return batches size and content when the order is deterministic.
TEST(DataLoaderTest, ChunkDataSetGetBatch) {
  // different prefetch count for testing.
  const size_t prefetch_counts[] = {1, 2, 3, 4};

  // different batch size for testing.
  const size_t batch_sizes[] = {5, 7};

  // test with/without worker threads
  const size_t dataloader_worker_counts[] = {0, 2};

  const size_t total_example_count = 35;
  DummyChunkDataReader data_reader;
  samplers::SequentialSampler sampler(0);

  // test functionality across epoch boundary
  const int epoch_count = 2;

  for (auto prefetch_count : prefetch_counts) {
    for (auto batch_size : batch_sizes) {
      for (auto dataloader_worker_count : dataloader_worker_counts) {
        datasets::SharedBatchDataset<datasets::ChunkDataset<
            DummyChunkDataReader,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>
            dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
                DummyChunkDataReader,
                samplers::SequentialSampler,
                samplers::SequentialSampler>>(
                data_reader,
                sampler,
                sampler,
                datasets::ChunkDatasetOptions(prefetch_count, batch_size));

        auto data_loader = torch::data::make_data_loader(
            dataset,
            DataLoaderOptions(batch_size).workers(dataloader_worker_count));

        for (int epoch_index = 0; epoch_index < epoch_count; ++epoch_index) {
          std::vector<bool> result(total_example_count, false);
          int iteration_count = 0;
          for (auto iterator = data_loader->begin();
               iterator != data_loader->end();
               ++iterator, ++iteration_count) {
            std::vector<int>& batch = *iterator;
            ASSERT_EQ(batch.size(), batch_size);

            // When prefetch_count is equal to 1 and no worker thread, the batch
            // order is deterministic. So we can verify elements in each batch.
            if (prefetch_count == 1 && dataloader_worker_count == 0) {
              for (size_t j = 0; j < batch_size; ++j) {
                ASSERT_EQ(batch[j], iteration_count * batch_size + j);
              }
            }
            for (size_t j = 0; j < batch_size; ++j) {
              result[batch[j]] = true;
            }
          }

          for (auto data : result) {
            ASSERT_EQ(data, true);
          }
        }
      }
    }
  }
}

TEST(DataLoaderTest, ChunkDataSetWithBatchSizeMismatch) {
  const size_t prefetch_count = 1;
  const size_t batch_size = 5;
  const size_t requested_batch_size = 6;

  DummyChunkDataReader data_reader;
  samplers::SequentialSampler sampler(0);

  datasets::SharedBatchDataset<datasets::ChunkDataset<
      DummyChunkDataReader,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          DummyChunkDataReader,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(prefetch_count, batch_size));

  auto data_loader = torch::data::make_data_loader(
      dataset,
      DataLoaderOptions(requested_batch_size).workers(0));

  std::string exception_msg =
      "The requested batch size does not match with the initialized batch "
      "size.\n The requested batch size is 6, while the dataset is created"
      " with batch size equal to 5";

  ASSERT_THROWS_WITH(*(data_loader->begin()), exception_msg);
}

TEST(DataLoaderTest, ChunkDataSetWithEmptyBatch) {
  struct DummyEmptyChunkDataReader
      : datasets::ChunkDataReader<std::vector<int>> {
   public:
    using BatchType = std::vector<int>;

    BatchType read_chunk(size_t chunk_index) override {
      return {};
    }

    size_t chunk_count() override {
      return 1;
    };

    void reset() override{};
  };

  const size_t prefetch_count = 1;
  const size_t batch_size = 5;
  DummyEmptyChunkDataReader data_reader;
  samplers::SequentialSampler sampler(0);

  datasets::SharedBatchDataset<datasets::ChunkDataset<
      DummyEmptyChunkDataReader,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          DummyEmptyChunkDataReader,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(prefetch_count, batch_size));

  auto data_loader = torch::data::make_data_loader(
      dataset, DataLoaderOptions(batch_size).workers(0));

  for (auto iterator = data_loader->begin(); iterator != data_loader->end();
       ++iterator) {
    ASSERT_EQ(iterator->size(), 0);
  }
}

TEST(DataLoaderTest, ChunkDataSetGetBatchWithUnevenBatchSize) {
  struct D : public datasets::ChunkDataReader<std::vector<int>> {
   public:
    using BatchType = std::vector<int>;

    BatchType read_chunk(size_t chunk_index) override {
      BatchType batch_data(10, 0);
      return batch_data;
    }

    size_t chunk_count() override {
      return 2;
    };

    void reset() override{};
  };

  const size_t batch_sizes[] = {17, 30};
  D data_reader;
  samplers::SequentialSampler sampler(0);

  for (auto batch_size : batch_sizes) {
    datasets::SharedBatchDataset<datasets::ChunkDataset<
        D,
        samplers::SequentialSampler,
        samplers::SequentialSampler>>
        dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
            D,
            samplers::SequentialSampler,
            samplers::SequentialSampler>>(
            data_reader,
            sampler,
            sampler,
            datasets::ChunkDatasetOptions(1, batch_size));

    auto data_loader = torch::data::make_data_loader(
        dataset, DataLoaderOptions(batch_size).workers(0));

    for (auto iterator = data_loader->begin(); iterator != data_loader->end();
         ++iterator) {
      std::vector<int> batch = *iterator;
      auto batch_size = batch.size();
      if (batch_size == 17) {
        ASSERT_TRUE(batch.size() == 17 || batch.size() == 3);
      }
      if (batch_size == 30) {
        ASSERT_TRUE(batch.size() == 20);
      }
    }
  }
}

TEST(DataLoaderTest, CanAccessChunkSamplerWithChunkDataSet) {
  const size_t prefetch_count = 2;
  const size_t batch_size = 5;

  DummyChunkDataReader data_reader;
  samplers::SequentialSampler sampler(0);
  datasets::SharedBatchDataset<datasets::ChunkDataset<
      DummyChunkDataReader,
      samplers::SequentialSampler,
      samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<datasets::ChunkDataset<
          DummyChunkDataReader,
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          data_reader,
          sampler,
          sampler,
          datasets::ChunkDatasetOptions(prefetch_count, batch_size));

  samplers::SequentialSampler& chunk_sampler = dataset->chunk_sampler();

  auto data_loader = torch::data::make_data_loader(
      dataset.map(transforms::BatchLambda<std::vector<int>, int>(
          [](std::vector<int> batch) {
            return std::accumulate(batch.begin(), batch.end(), 0);
          })),
      DataLoaderOptions(batch_size).workers(0));

  // before we start, the index should be 0.
  ASSERT_EQ(chunk_sampler.index(), 0);

  size_t sum = 0;
  for (auto iterator = data_loader->begin(); iterator != data_loader->end();
       ++iterator) {
    sum += *iterator;
  }
  ASSERT_EQ(sum, 595); // sum([0, 35))
  // 3 chunks, and when exhausted the value is already incremented.
  ASSERT_EQ(chunk_sampler.index(), 3);
}
