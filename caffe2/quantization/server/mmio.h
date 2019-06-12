#pragma once

#include <cstdio>
#include <set>
#include <string>
#include <type_traits>

namespace caffe2 {

template <typename T>
void StoreMatrixInMatrixMarketFormat(
    int m, int n, const T *a, const std::string& matrix_name) {
  using namespace std;
  static_assert(
      is_integral<T>::value,
      "StoreMatrixInMatrixMarket only works with integer types");

  static set<string> dumped_matrix_names;

  string name(matrix_name);
  string::size_type pos = name.rfind('/');
  if (pos != string::npos) {
    name = name.substr(pos + 1);
  }
  if (dumped_matrix_names.find(name) == dumped_matrix_names.end()) {
    dumped_matrix_names.insert(name);

    FILE *fp = fopen((name + ".mtx").c_str(), "w");
    if (!fp) {
      return;
    }

    fprintf(fp, "%%%%MatrixMarket matrix array integer general\n");
    fprintf(fp, "%d %d\n", m, n);
    // matrix market array format uses column-major order
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < m; ++i) {
        fprintf(fp, "%d\n", a[j * m + i]);
      }
    }

    fclose(fp);
  }
}

} // namespace caffe2
