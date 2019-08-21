#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

#include <immintrin.h>

#include "embedding_lookup.h"
#include "embedding_lookup_idx.h"

using namespace std;

namespace {
template <typename T>
void llc_flush(std::vector<T>& v) {
  constexpr int CACHE_LINE_SIZE = 64;
  for (int i = 0; i < v.size(); i += CACHE_LINE_SIZE / sizeof(T)) {
    _mm_clflush(&v[i]);
  }
}
} // anonymous namespace

int main(int argc, const char* argv[]) {
  // int batch_size = 100;
  // int num_unique_ids = 500;
  // int embedding_dim = 48;
  // int average_len = 100;

  int batch_size = 5;
  int num_unique_ids = 20;
  int embedding_dim = 4;
  int average_len = 6;

  if (argc > 1) {
    batch_size = atoi(argv[1]);
  }
  if (argc > 2) {
    num_unique_ids = atoi(argv[2]);
  }
  if (argc > 3) {
    embedding_dim = atoi(argv[3]);
  }
  if (argc > 4) {
    average_len = atoi(argv[4]);
  }

  cout << "batch_size: " << batch_size << ";"
       << "num_unique_ids: " << num_unique_ids << ";"
       << "embedding_dim: " << embedding_dim << ";"
       << "average_len: " << average_len << ";" << endl;

  // Create embedding table
  vector<float> embedding_table(num_unique_ids * embedding_dim);
  default_random_engine generator;
  normal_distribution<float> embedding_distribution;
  for (int i = 0; i < embedding_table.size(); ++i) {
    embedding_table[i] = embedding_distribution(generator);
  }

  cout << "embedding_table: " << endl;
  for (int i = 0; i < num_unique_ids; i++) {
    for (int j = 0; j < embedding_dim; j++) {
      cout << embedding_table[i * embedding_dim + j] << " ";
    }
    cout << endl;
  }

  // Generate lengths
  uniform_int_distribution<int> length_distribution(1, 2 * average_len + 1);
  vector<int> lengths(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    lengths[i] = length_distribution(generator);
  }

  cout << "lengths: " << endl;
  for (int i = 0; i < batch_size; ++i) {
    cout << lengths[i] << " ";
  }
  cout << endl;

  // Calculate offsets
  vector<int> offsets(batch_size);
  int cumsum = 0;
  for (int i = 0; i < batch_size; ++i) {
    offsets[i] = cumsum;
    cumsum += lengths[i];
  }

  cout << "offsets: " << endl;
  for (int i = 0; i < batch_size; ++i) {
    cout << offsets[i] << " ";
  }
  cout << endl;

  // Compute the number of indices
  int lengths_sum = accumulate(lengths.begin(), lengths.end(), 0);
  cout << "lengths_sum " << lengths_sum << endl;

  // Generate indices
  vector<int64_t> indices;
  vector<int> container(num_unique_ids);
  map<int64_t, set<int>> dedup_map; // index -> set(output index)
  for (int i = 0; i < batch_size; ++i) {
    iota(container.begin(), container.end(), 0);
    random_shuffle(container.begin(), container.end());
    copy(
        container.begin(),
        container.begin() + lengths[i],
        back_inserter(indices));
  }

  cout << "indices: " << endl;
  for (int i = 0; i < indices.size(); i++) {
    cout << indices[i] << " ";
  }
  cout << endl;

  // Generate weights
  vector<float> weights(lengths_sum);
  for (int i = 0; i < lengths_sum; ++i) {
    weights[i] = embedding_distribution(generator);
  }

  cout << "weights: " << endl;
  for (int i = 0; i < lengths_sum; ++i) {
    cout << weights[i] << " ";
  }
  cout << endl;

  vector<float> output_sls_ref(batch_size * embedding_dim);
  vector<float> output_slws_ref(output_sls_ref.size()),
      output_sls(output_sls_ref.size()), output_slws(output_sls_ref.size());

  vector<char> llc(64L * 1024L * 1024L, 1.0);

  chrono::time_point<chrono::system_clock> t_begin, t_end;
  double t;

  constexpr int NUM_WARMUP = 4;
  constexpr int NUM_ITER = 64;
  // Only counts the number of bytes for reading embedding table and ignore
  // others. Should be good enough as long as embdding_dim is big enough.
  double bytes = static_cast<double>(NUM_ITER) * lengths_sum * embedding_dim *
      sizeof(float);

  // Baseline
  for (bool has_weight : {false, true}) {
    vector<float>& output = has_weight ? output_slws_ref : output_sls_ref;

    for (bool flush_cache : {false, true}) {
      t = 0;
      for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
        if (flush_cache) {
          llc_flush(embedding_table);
          llc_flush(indices);
          llc_flush(lengths);
          llc_flush(weights);
          llc_flush(output);
        }

        t_begin = chrono::system_clock::now();

        caffe2::EmbeddingLookup(
            embedding_dim /* block_size */,
            batch_size /* output_size */,
            lengths_sum /* index_size */,
            num_unique_ids /* data_size */,
            embedding_table.data(), /* data_size x embedding_dim */
            indices.data(), /* data_size */
            lengths.data(), /* output_size */
            has_weight ? weights.data() : nullptr,
            nullptr,
            false,
            output.data()); /* output_size */

        t_end = chrono::system_clock::now();
        if (i >= NUM_WARMUP) {
          t += chrono::duration<double>(t_end - t_begin).count();
        }
      }

      if (has_weight) {
        cout << "SLWS ";
      } else {
        cout << "SLS ";
      }
      if (flush_cache) {
        cout << " cache_flush ";
      }

      cout << bytes / 1e9 / t << " GB/s" << endl;
    } // flush_cache
  } // has_weight

  // Change to Offset interface
  for (bool has_weight : {false, true}) {
    vector<float>& output = has_weight ? output_slws : output_sls;

    fill(output.begin(), output.end(), 0.0f);
    for (bool flush_cache : {false, true}) {
      t = 0;
      for (int i = 0; i < NUM_WARMUP + NUM_ITER; ++i) {
        if (flush_cache) {
          llc_flush(embedding_table);
          llc_flush(indices);
          llc_flush(lengths);
          llc_flush(weights);
          llc_flush(output);
        }

        t_begin = chrono::system_clock::now();

        // Calculating the Offsets!

        caffe2::EmbeddingLookupIdx(
            embedding_dim /* block_size */,
            batch_size /* output_size */,
            lengths_sum /* index_size */,
            num_unique_ids /* data_size */,
            embedding_table.data(), /* data_size x embedding_dim */
            indices.data(), /* data_size */
            offsets.data(), /* output_size */
            has_weight ? weights.data() : nullptr,
            nullptr,
            false,
            output.data()); /* output_size */

        t_end = chrono::system_clock::now();
        if (i >= NUM_WARMUP) {
          t += chrono::duration<double>(t_end - t_begin).count();
        }
      }

      // Check correctness
      if (!flush_cache) {
        vector<float>& output_ref =
            has_weight ? output_slws_ref : output_sls_ref;
        for (int i = 0; i < output.size(); ++i) {
          assert(fabs(output[i] - output_ref[i]) < fabs(output_ref[i]) * 1e-3);
          if (fabs(output[i] - output_ref[i]) >= fabs(output_ref[i]) * 1e-3) {
            cout << i << " " << output[i] << " " << output_ref[i] << endl;
          }
        }
      }

      if (has_weight) {
        cout << "SLWS ";
      } else {
        cout << "SLS ";
      }
      if (flush_cache) {
        cout << " cache_flush ";
      }

      cout << bytes / 1e9 / t << " GB/s" << endl;
    } // flush_cache
  } // has_weight

  return 0;
}
