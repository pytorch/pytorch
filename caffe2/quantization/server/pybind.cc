#include "caffe2_dnnlowp_utils.h"
#include "activation_distribution_observer.h"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(dnnlowp_pybind11, m) {
  using namespace std;
  using namespace caffe2;

  m.def(
      "ClearNetObservers",
      []() {
        ClearGlobalNetObservers();
      });

  m.def(
      "ObserveMinMaxOfOutput",
      [](const string& min_max_file_name, int dump_freq) {
        AddGlobalNetObserverCreator(
          [dump_freq, min_max_file_name](NetBase* net) {
            return make_unique<OutputMinMaxNetObserver>(
              net, min_max_file_name, dump_freq);
          });
      },
      pybind11::arg("min_max_file_name"),
      pybind11::arg("dump_freq") = -1);

  m.def(
      "ObserveHistogramOfOutput",
      [](const string& out_file_name, int dump_freq) {
        AddGlobalNetObserverCreator(
          [out_file_name, dump_freq](NetBase* net) {
            return make_unique<HistogramNetObserver>(
              net, out_file_name, 2048, dump_freq);
          });
      },
      pybind11::arg("out_file_name"),
      pybind11::arg("dump_freq") = -1);

  m.def(
      "RegisterQuantizationParams",
      [](const string& min_max_file_name,
         bool is_weight,
         const string& qparams_output_file_name) {
        AddGlobalNetObserverCreator([min_max_file_name,
                                     is_weight,
                                     qparams_output_file_name](NetBase* net) {
          return make_unique<RegisterQuantizationParamsNetObserver>(
              net, min_max_file_name, is_weight, qparams_output_file_name);
        });
      },
      pybind11::arg("min_max_file_name"),
      pybind11::arg("is_weight") = false,
      pybind11::arg("qparams_output_file_name") = "");

  m.def(
      "RegisterQuantizationParamsWithHistogram",
      [](const string& histogram_file_name,
         bool is_weight,
         const string& qparams_output_file_name) {
        AddGlobalNetObserverCreator([histogram_file_name,
                                     is_weight,
                                     qparams_output_file_name](NetBase* net) {
          return make_unique<
              RegisterQuantizationParamsWithHistogramNetObserver>(
              net, histogram_file_name, is_weight, qparams_output_file_name);
        });
      },
      pybind11::arg("histogram_file_name"),
      pybind11::arg("is_weight") = false,
      pybind11::arg("qparams_output_file_name") = "");

  m.def(
      "AddScaleZeroOffsetArgumentsWithHistogram",
      [](const pybind11::bytes& net_def_bytes,
         const string& histogram_file_name) {
        NetDef def;
        CAFFE_ENFORCE(ParseProtoFromLargeString(
            net_def_bytes.cast<string>(), &def));
        pybind11::gil_scoped_release g;

        string protob;
        auto transformed_net =
          dnnlowp::AddScaleZeroOffsetArgumentsWithHistogram(
            def, histogram_file_name);

        CAFFE_ENFORCE(transformed_net.SerializeToString(&protob));
        return pybind11::bytes(protob);
      });
}
