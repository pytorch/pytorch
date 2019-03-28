#include <pybind11/pybind11.h>
#include "activation_distribution_observer.h"
#include "caffe2_dnnlowp_utils.h"

namespace caffe2 {
namespace python {
// defined in caffe2/python/pybind_state.cc
Workspace* GetCurrentWorkspace();
} // namespace python
} // namespace caffe2

PYBIND11_MODULE(dnnlowp_pybind11, m) {
  using namespace std;
  using namespace caffe2;

  m.def("ClearNetObservers", []() { ClearGlobalNetObservers(); });

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
      [](const string& out_file_name, int dump_freq, bool mul_nets) {
        AddGlobalNetObserverCreator(
            [out_file_name, dump_freq, mul_nets](NetBase* net) {
              return make_unique<HistogramNetObserver>(
                  net, out_file_name, 2048, dump_freq, mul_nets);
            });
      },
      pybind11::arg("out_file_name"),
      pybind11::arg("dump_freq") = -1,
      pybind11::arg("mul_nets") = false);

  m.def(
      "AddHistogramObserver",
      [](const string& net_name,
         const string& out_file_name,
         int dump_freq,
         bool mul_nets) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        CAFFE_ENFORCE(
            gWorkspace->GetNet(net_name), "Can't find net ", net_name);
        pybind11::gil_scoped_release g;

        NetBase* net = gWorkspace->GetNet(net_name);
        const Observable<NetBase>::Observer* observer = nullptr;

        observer = net->AttachObserver(make_unique<HistogramNetObserver>(
            net, out_file_name, 2048, dump_freq, mul_nets));

        CAFFE_ENFORCE(observer != nullptr);
        return pybind11::cast(observer);
      },
      pybind11::arg("net_name"),
      pybind11::arg("out_file_name"),
      pybind11::arg("dump_freq") = -1,
      pybind11::arg("mul_nets") = false);

  m.def(
      "ChooseQuantizationParams",
      [](const std::string& blob_name) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        pybind11::gil_scoped_release g;

        const auto* blob = gWorkspace->GetBlob(blob_name);
        if (blob == nullptr) {
          LOG(WARNING) << "Can't find blob " << blob_name;
        } else if (BlobIsTensorType(*blob, CPU)) {
          LOG(WARNING) << "Blob " << blob_name << " is not a tensor";
        } else {
          const auto& tensor = blob->template Get<Tensor>();
          if (tensor.IsType<float>()) {
            dnnlowp::QuantizationFactory* qfactory =
                dnnlowp::QuantizationFactory::GetDefaultInstance();
            dnnlowp::TensorQuantizationParams qparams =
                qfactory->ChooseQuantizationParams(
                    tensor.data<float>(), tensor.size(), true /*weight*/);
            return std::tuple<float, int>(qparams.scale, qparams.zero_point);
          } else {
            LOG(WARNING) << "Blob " << blob_name << " is not a float tensor";
          }
        }
        return std::tuple<float, int>(1.0, 0);
      },
      pybind11::arg("blob_name"));

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
      "AddRegisterQuantizationParamsWithHistogramObserver",
      [](const string& net_name,
         const string& histogram_file_name,
         int is_weight,
         const string& qparams_output_file_name) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        CAFFE_ENFORCE(
            gWorkspace->GetNet(net_name), "Can't find net ", net_name);
        pybind11::gil_scoped_release g;

        NetBase* net = gWorkspace->GetNet(net_name);
        const Observable<NetBase>::Observer* observer = nullptr;

        observer = net->AttachObserver(
            make_unique<RegisterQuantizationParamsWithHistogramNetObserver>(
                net, histogram_file_name, is_weight, qparams_output_file_name));

        CAFFE_ENFORCE(observer != nullptr);
        return pybind11::cast(observer);
      },
      pybind11::arg("net_name"),
      pybind11::arg("histogram_file_name"),
      pybind11::arg("is_weight") = false,
      pybind11::arg("qparams_output_file_name") = "");

  m.def(
      "AddScaleZeroOffsetArgumentsWithHistogram",
      [](const pybind11::bytes& net_def_bytes,
         const string& histogram_file_name) {
        NetDef def;
        CAFFE_ENFORCE(
            ParseProtoFromLargeString(net_def_bytes.cast<string>(), &def));
        pybind11::gil_scoped_release g;

        string protob;
        auto transformed_net =
            dnnlowp::AddScaleZeroOffsetArgumentsWithHistogram(
                def, histogram_file_name);

        CAFFE_ENFORCE(transformed_net.SerializeToString(&protob));
        return pybind11::bytes(protob);
      });
}
