#include <fbgemm/FbgemmFP16.h>
#include <fbgemm/Utils.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "activation_distribution_observer.h"
#include "caffe2/opt/custom/fakefp16_transform.h"
#include "caffe2/quantization/server/fbgemm_pack_blob.h"
#include "caffe2_dnnlowp_utils.h"
#include "quantization_error_minimization.h"

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
      [](const string& min_max_file_name, int dump_freq, string delimiter) {
        AddGlobalNetObserverCreator(
            [dump_freq, min_max_file_name, delimiter](NetBase* net) {
              return make_unique<OutputMinMaxNetObserver>(
                  net, min_max_file_name, dump_freq, delimiter);
            });
      },
      pybind11::arg("min_max_file_name"),
      pybind11::arg("dump_freq") = -1,
      pybind11::arg("delimiter") = " ");

  m.def(
      "ObserveHistogramOfOutput",
      [](const string& out_file_name,
         int dump_freq,
         bool mul_nets,
         string op_filter,
         string delimiter) {
        AddGlobalNetObserverCreator(
            [out_file_name, dump_freq, mul_nets, op_filter, delimiter](
                NetBase* net) {
              return make_unique<HistogramNetObserver>(
                  net,
                  out_file_name,
                  2048,
                  dump_freq,
                  mul_nets,
                  op_filter,
                  delimiter);
            });
      },
      pybind11::arg("out_file_name"),
      pybind11::arg("dump_freq") = -1,
      pybind11::arg("mul_nets") = false,
      pybind11::arg("op_filter") = "",
      pybind11::arg("delimiter") = " ");

  m.def(
      "DumpHistogramFile",
      [](Observable<NetBase>::Observer* ob) {
        HistogramNetObserver* hist_ob =
            dynamic_cast_if_rtti<HistogramNetObserver*>(ob);
        hist_ob->DumpHistogramFile();
      },
      pybind11::arg("ob"));

  m.def(
      "AddHistogramObserver",
      [](const string& net_name,
         const string& out_file_name,
         int dump_freq,
         bool mul_nets,
         string delimiter) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        CAFFE_ENFORCE(
            gWorkspace->GetNet(net_name), "Can't find net ", net_name);
        pybind11::gil_scoped_release g;

        NetBase* net = gWorkspace->GetNet(net_name);
        const Observable<NetBase>::Observer* observer = nullptr;

        observer = net->AttachObserver(make_unique<HistogramNetObserver>(
            net, out_file_name, 2048, dump_freq, mul_nets, "", delimiter));

        CAFFE_ENFORCE(observer != nullptr);
        return pybind11::cast(observer);
      },
      pybind11::arg("net_name"),
      pybind11::arg("out_file_name"),
      pybind11::arg("dump_freq") = -1,
      pybind11::arg("mul_nets") = false,
      pybind11::arg("delimiter") = " ");

  m.def(
      "AddOutputColumnMaxHistogramObserver",
      [](const string& net_name,
         const string& out_file_name,
         const std::vector<std::string>& observe_column_max_for_blobs,
         int dump_freq,
         bool mul_nets,
         string delimiter) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        CAFFE_ENFORCE(
            gWorkspace->GetNet(net_name), "Can't find net ", net_name);
        pybind11::gil_scoped_release g;

        NetBase* net = gWorkspace->GetNet(net_name);
        const Observable<NetBase>::Observer* observer = nullptr;

        observer = net->AttachObserver(
            make_unique<OutputColumnMaxHistogramNetObserver>(
                net,
                out_file_name,
                observe_column_max_for_blobs,
                2048,
                dump_freq,
                mul_nets,
                delimiter));

        CAFFE_ENFORCE(observer != nullptr);
        return pybind11::cast(observer);
      },
      pybind11::arg("net_name"),
      pybind11::arg("out_file_name"),
      pybind11::arg("observe_column_max_for_blobs"),
      pybind11::arg("dump_freq") = -1,
      pybind11::arg("mul_nets") = false,
      pybind11::arg("delimiter") = " ");

  m.def(
      "ChooseQuantizationParams",
      [](const std::string& blob_name) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        pybind11::gil_scoped_release g;

        const auto* blob = gWorkspace->GetBlob(blob_name);
        if (blob == nullptr) {
          LOG(WARNING) << "Can't find blob " << blob_name;
        } else if (!BlobIsTensorType(*blob, CPU)) {
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

  pybind11::class_<dnnlowp::TensorQuantizationParams>(m, "QueryTensorQparam")
      .def(pybind11::init<float, std::int32_t, int>())
      .def_property_readonly(
          "scale",
          [](dnnlowp::TensorQuantizationParams& qparam) {
            return qparam.scale;
          })
      .def_property_readonly(
          "zero_point",
          [](dnnlowp::TensorQuantizationParams& qparam) {
            return qparam.zero_point;
          })
      .def_property_readonly(
          "precision",
          [](dnnlowp::TensorQuantizationParams& qparam) {
            return qparam.precision;
          })
      .def_property_readonly(
          "min",
          [](dnnlowp::TensorQuantizationParams& qparam) {
            return qparam.Min();
          })
      .def_property_readonly(
          "max", [](dnnlowp::TensorQuantizationParams& qparam) {
            return qparam.Max();
          });
  m.def("get_fakefp16_mapping", [](bool use_fp16_acc, bool use_nnpi) {
    return caffe2::opt::getFakeFp16OpMapping(use_fp16_acc, use_nnpi);
  });
  m.def(
      "ChooseStaticQuantizationParams",
      [](float min,
         float max,
         const std::vector<uint64_t>& bins,
         bool preserve_sparsity,
         int precision,
         const std::string& quant_scheme,
         float p99_threshold,
         bool is_weight) {
        dnnlowp::Histogram hist = dnnlowp::Histogram(min, max, bins);

        dnnlowp::QuantizationFactory::QuantizationKind quant_kind =
            dnnlowp::QuantizationFactory::MIN_MAX_QUANTIZATION;
        if (quant_scheme.compare("L2_MIN_QUANTIZATION") == 0) {
          quant_kind = dnnlowp::QuantizationFactory::L2_MIN_QUANTIZATION;
        } else if (quant_scheme.compare("L2_MIN_QUANTIZATION_APPROX") == 0) {
          quant_kind = dnnlowp::QuantizationFactory::L2_MIN_QUANTIZATION_APPROX;
        } else if (quant_scheme.compare("KL_MIN_QUANTIZATION") == 0) {
          quant_kind = dnnlowp::QuantizationFactory::KL_MIN_QUANTIZATION;
        } else if (quant_scheme.compare("P99_QUANTIZATION") == 0) {
          quant_kind = dnnlowp::QuantizationFactory::P99_QUANTIZATION;
        } else if (quant_scheme.compare("L1_MIN_QUANTIZATION") == 0) {
          quant_kind = dnnlowp::QuantizationFactory::L1_MIN_QUANTIZATION;
        } else {
          LOG(INFO) << "Using DNNLOWP default MIN_MAX_QUANTIZATION";
        }
        dnnlowp::QuantizationFactory* qfactory =
            dnnlowp::QuantizationFactory::GetDefaultInstance();
        if (is_weight) {
          qfactory->SetWeightP99Threshold(p99_threshold);
        } else {
          qfactory->SetActivationP99Threshold(p99_threshold);
        }
        return qfactory->ChooseQuantizationParams(
            hist, quant_kind, precision, preserve_sparsity, is_weight);
      },
      pybind11::arg("min"),
      pybind11::arg("max"),
      pybind11::arg("bins"),
      pybind11::arg("preserve_sparsity") = true,
      pybind11::arg("precision") = 8,
      pybind11::arg("quant_scheme") = "min_max",
      pybind11::arg("p99_threshold") = 0.99,
      pybind11::arg("is_weight") = false);
  m.def(
      "ObserveFp16FCPackedWeights",
      [](const string& blob_name, const string& weights_out_file) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        const auto* blob = gWorkspace->GetBlob(blob_name);
        CAFFE_ENFORCE(blob, "Can't find blob ", blob_name);
        fbgemm::PackedGemmMatrixFP16* packedGemmMatrixPtr =
            blob->template Get<unique_ptr<fbgemm::PackedGemmMatrixFP16>>()
                .get();
        uint64_t nrow = packedGemmMatrixPtr->numRows();
        uint64_t ncol = packedGemmMatrixPtr->numCols();
        uint64_t size = nrow * ncol;
        fbgemm::float16* unpacked_mat_ptr = nullptr;
        vector<fbgemm::float16> unpacked_mat;

        if (!packedGemmMatrixPtr->packed()) {
          unpacked_mat_ptr = packedGemmMatrixPtr->pmat();
        } else {
          unpacked_mat.resize(size);
          packedGemmMatrixPtr->unpack(
              unpacked_mat.data(), fbgemm::matrix_op_t::Transpose);
          unpacked_mat_ptr = unpacked_mat.data();
        }
        ofstream fout;
        fout.open(weights_out_file);
        if (!fout) {
          LOG(WARNING) << "Can't open output file to dump fp16 weights "
                       << weights_out_file;
          return;
        }
        for (int i = 0; i < nrow; ++i) {
          for (int j = 0; j < ncol; ++j) {
            if (j > 0) {
              fout << " ";
            }
            fout << fbgemm::cpu_half2float(unpacked_mat_ptr[i + nrow * j]);
          }
          fout << endl;
        }
        LOG(INFO) << "Written unpacked blob " << blob_name << " to "
                  << weights_out_file;
      },
      pybind11::arg("blob_name"),
      pybind11::arg("weights_out_file"));
  m.def(
      "ObserveInt8FCPackedWeights",
      [](const string& blob_name, const string& weights_out_file) {
        Workspace* gWorkspace = caffe2::python::GetCurrentWorkspace();
        CAFFE_ENFORCE(gWorkspace);
        const auto* blob = gWorkspace->GetBlob(blob_name);
        if (blob == nullptr) {
          LOG(WARNING) << "Can't find blob " << blob_name;
          return;
        }
        const Int8FCDNNLowPPackedWeightBlob& packedInt8Blob =
            blob->template Get<Int8FCDNNLowPPackedWeightBlob>();
        auto& qparams = packedInt8Blob.qparams;
        auto& int8_tensor = packedInt8Blob.original_tensor;

        auto shape = int8_tensor.sizes();

        ofstream fout;
        fout.open(weights_out_file);
        if (!fout) {
          LOG(WARNING) << "Can't open output file to dump int8 weights "
                       << weights_out_file;
          return;
        }
        for (int i = 0; i < qparams.size(); ++i) {
          if (i > 0) {
            fout << " ";
          }
          fout << to_string(qparams[i].scale) << " "
               << to_string(qparams[i].zero_point);
        }
        fout << endl;
        int8_t* int8_data = int8_tensor.data<int8_t>();
        for (int i = 0; i < shape[0]; ++i) {
          for (int j = 0; j < shape[1]; ++j) {
            if (j > 0) {
              fout << " ";
            }
            fout << to_string(int8_data[i * shape[1] + j]);
          }
          fout << endl;
        }
        LOG(INFO) << "Written int8 qparams and weights for " << blob_name
                  << " to " << weights_out_file;
      },
      pybind11::arg("blob_name"),
      pybind11::arg("weights_out_file"));
}
