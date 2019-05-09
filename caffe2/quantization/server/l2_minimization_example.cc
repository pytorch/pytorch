#include "caffe2/core/logging.h"
#include "l2_minimization.h"

#include <fstream>
#include <iostream>
#include <sstream>

using namespace std;
using namespace dnnlowp;

int main(int argc, const char* argv[]) {
  if (argc < 3) {
    cerr << "Usage: " << argv[0]
         << " in_file out_file [preserve_sparsity] [precision]" << endl;
    return -1;
  }

  ifstream in(argv[1]);
  ofstream out(argv[2]);
  bool preserve_sparsity = argc >= 4 ? atoi(argv[3]) : false;
  int precision = argc >= 5 ? atoi(argv[4]) : 8;

  vector<tuple<int, string, int, string>> infos;
  vector<Histogram> hists;

  string line;
  while (getline(in, line)) {
    istringstream ist(line);

    int op_index, output_index;
    string op_type, tensor_name;
    float min, max;
    int nbins;

    ist >> op_index >> op_type >> output_index >> tensor_name >> min >> max >>
        nbins;
    infos.push_back(tuple<int, string, int, string>(
        op_index, op_type, output_index, tensor_name));

    vector<uint64_t> bins;
    for (int i = 0; i < nbins; ++i) {
      uint64_t cnt;
      ist >> cnt;
      bins.push_back(cnt);
    }
    assert(bins.size() == nbins);

    Histogram hist = Histogram(min, max, bins);
    hists.emplace_back(min, max, bins);
  }

  vector<TensorQuantizationParams> qparams(hists.size());

  for (int i = 0; i < hists.size(); ++i) {
    qparams[i] = L2ErrorMinimization().ChooseQuantizationParams(
        hists[i], preserve_sparsity, precision);
  }

  for (int i = 0; i < qparams.size(); ++i) {
    VLOG(2) << std::get<2>(infos[i]);
    out << std::get<0>(infos[i]) << " " << std::get<1>(infos[i]) << " "
        << std::get<2>(infos[i]) << " " << std::get<3>(infos[i]) << " "
        << qparams[i].Min() << " " << qparams[i].Max() << endl;
  }

  return 0;
}
