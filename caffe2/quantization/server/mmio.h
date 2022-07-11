#pragma once

#include <cstdio>
#include <set>
#include <string>
#include <type_traits>

namespace caffe2 {

template <typename T>
void StoreMatrixInMatrixMarketFormat(
    int m,
    int n,
    const T* a,
    const std::string& matrix_name) {
  using namespace std;
  static set<string> dumped_matrix_names;

  string name(matrix_name);
  string::size_type pos = name.rfind('/');
  if (pos != string::npos) {
    name = name.substr(pos + 1);
  }
  if (dumped_matrix_names.find(name) == dumped_matrix_names.end()) {
    dumped_matrix_names.insert(name);

    FILE* fp = fopen((matrix_name + ".mtx").c_str(), "w");
    if (!fp) {
      return;
    }

    if (is_integral<T>::value) {
      fprintf(fp, "%%%%MatrixMarket matrix array integer general\n");
    } else {
      fprintf(fp, "%%%%MatrixMarket matrix array real general\n");
    }
    fprintf(fp, "%d %d\n", m, n);
    // matrix market array format uses column-major order
    for (const auto j : c10::irange(n)) {
      for (const auto i : c10::irange(m)) {
        if (is_integral<T>::value) {
          // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
          fprintf(fp, "%d\n", static_cast<int>(a[j * m + i]));
        } else {
          fprintf(fp, "%f\n", static_cast<float>(a[j * m + i]));
        }
      }
    }

    fclose(fp);
  }
}

} // namespace caffe2
