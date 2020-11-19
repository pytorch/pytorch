#include <iostream>
#include <regex>
#include <unordered_map>

#include <torch/csrc/jit/api/module.h>
#include "./address.cpp"

#include<typeinfo>

using namespace std;
using namespace torch::jit;

namespace fun {
#define SIZE 64
int layer_num;
int layer_num_bf;
char layer_type[SIZE];
int num[3] = {0,0,0};
char layer_type_bf[SIZE];
torch::jit::Node* all_node_back = 0;
torch::jit::Value* GetAttrValue = 0;
torch::jit::Node* conv_node_back = 0;

class Allocator {
 private:
  uint64_t dram_base = 0;
  const uint64_t dram_capacity = ((uint64_t)1) << 32;

 public:
  Allocator() {
  }
  uint64_t dram_allocate(uint64_t n) {
    uint64_t new_dram_base = dram_base + n;
    TORCH_CHECK(new_dram_base <= dram_capacity, "dram_capacity not enough");
    uint64_t dram_base_bak = dram_base;
    dram_base = new_dram_base;
    return dram_base_bak;
  }
};

struct Conv2dParameter {
  int in_channels;
  int out_channels;
  int kernel_size_x;
  int kernel_size_y;
  int stride_x;
  int stride_y;
  int dilation_x;
  int dilation_y;
  int weight;
  int feature_map_size_x;
  int feature_map_size_y;
  bool transposed;
};

struct BatchNormParameter {
  bool  flag;
  int   dimen;
};


struct LinearParameter {
  int in_features_x;
  int in_features_y;
  int out_features_x;
  int out_features_y;
};

struct Pool2dParameter {
  int kernel_size_x;
  int kernel_size_y;
  int stride_x;
  int stride_y;
};

struct NNKnifeResult {
  int of_chiplet;
  int Y2;
  int X2;
  int K2;
  int Kp;
  int Yp;
  int Kc;
  int act_tile_hor;
  int act_tile_chl;
  int act_str_line;
  int act_str_chl;
  int act_tile_ver;
};

NNKnifeResult NNKnife() {
  NNKnifeResult result;
  result.of_chiplet = 4;
  result.Y2 = 2;
  result.X2 = 4;
  result.K2 = 1;
  result.Kc = 4;
  result.Kp = 1;
  result.Yp = 4;
  return result;
}

LinearParameter param_fc;
Conv2dParameter param_conv;
BatchNormParameter param_bn;

bool is_module(torch::jit::Node* node, string str) {
  TORCH_CHECK(
      node->kind() == prim::CallMethod,
      "Kind of node to be prim::CallMethod, but got ",
      string(node->kind().toUnqualString()));

  auto type = node->inputs()[0]->type()->cast<c10::ClassType>();
  if (type && type->name()) {
    static std::regex mangle_re("\\.___torch_mangle_\\d+");
    auto qualified_name =
        std::regex_replace(type->name()->qualifiedName(), mangle_re, "");
    return qualified_name == str;
  }

  return false;
}

class Compiler {
 private:
  unordered_map<string, Module> children = {};
  unordered_map<torch::jit::Value*, uint64_t> address = {};
  Module module;
  Allocator* allocator = new Allocator();

 public:
  Compiler(Module module) : module(module) {
    for (const NameModule& s : module.named_children()) {
      children[s.name] = s.value;
    }
  }

  BatchNormParameter parseBatchNorm(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.batchnorm.BatchNorm2d"),
        "node to be BatchNorm2d");

    BatchNormParameter param;
    auto value = node->inputs()[1];
	
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto size = pt->sizes().concrete_sizes(); 
    if (size.has_value()) {
	auto sizes = pt->sizes().concrete_sizes().value();
        param.dimen = sizes[1];
    } else {
        TORCH_CHECK(
                node->kind() == prim::CallMethod,
                "Kind of node to be prim::CallMethod, but got ",
                string(node->kind().toUnqualString()));
        const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
        auto child_graph = children[child_name].get_method("forward").graph();
        auto children_output = child_graph->outputs()[0]->type()->cast<TensorType>();
	auto sizes = children_output->sizes().concrete_sizes().value();
        param.dimen = sizes[1];
    }

    return param;
  }

  Conv2dParameter parseConv2d(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.conv.Conv2d") ||
        is_module(node, "__torch__.torch.nn.modules.conv.ConvTranspose2d"),
        "node to be Conv2d");

    Conv2dParameter param;

    auto value = node->inputs()[1];
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);

    auto size = pt->sizes().concrete_sizes(); 
    if (size.has_value()) {
	auto sizes = pt->sizes().concrete_sizes().value();
        param.in_channels = sizes[1];
    } else {
        TORCH_CHECK(
                node->kind() == prim::CallMethod,
                "Kind of node to be prim::CallMethod, but got ",
                string(node->kind().toUnqualString()));
        const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
        auto child_graph = children[child_name].get_method("forward").graph();
        auto children_output = child_graph->outputs()[0]->type()->cast<TensorType>();
	auto sizes = children_output->sizes().concrete_sizes().value();
        param.in_channels = sizes[1];
    }

    param.out_channels = shape(node->output())[1];

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    for (auto&& i : children[child_name].named_parameters(false)) {
      if (i.name == "weight") {
        param.kernel_size_x = i.value.sizes()[2];
        param.kernel_size_y = i.value.sizes()[3];
        break;
      }
    }

    auto child_graph = children[child_name].get_method("forward").graph();
    auto _convolution_node = child_graph->outputs()[0]->node();

    auto dilation_list = _convolution_node->inputs()[5]->node()->inputs();
    param.dilation_x = dilation_list[0]->node()->i(attr::value);
    param.dilation_y = dilation_list[1]->node()->i(attr::value);

    auto stride_list = _convolution_node->inputs()[3]->node()->inputs();
    param.stride_x = stride_list[0]->node()->i(attr::value);
    param.stride_y = stride_list[1]->node()->i(attr::value);

    param.transposed = _convolution_node->inputs()[6]->node()->i(attr::value);

    return param;
  }

  Pool2dParameter parsePool2d(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d") ||
        is_module(node, "__torch__.torch.nn.modules.pooling.AvgPool2d"),
        "node to be MaxPool2d or AvgPool2d");

    Pool2dParameter param;

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    auto child_graph = children[child_name].get_method("forward").graph();
    auto pool2d_node = child_graph->outputs()[0]->node();
    auto kernel_size_list = pool2d_node->inputs()[1]->node()->inputs();

    param.kernel_size_x = kernel_size_list[0]->node()->i(attr::value);
    param.kernel_size_y = kernel_size_list[1]->node()->i(attr::value);

    auto stride_list = pool2d_node->inputs()[2]->node()->inputs();
    param.stride_x = stride_list[0]->node()->i(attr::value);
    param.stride_y = stride_list[1]->node()->i(attr::value);

    return param;
  }

  LinearParameter parseLinear(torch::jit::Node* node) {
    unordered_map<string, Module> grand_children = {};
    const std::string& linear_name = node->s(attr::name);
    auto sequential_module = children["classifier"];
    for (const NameModule& s : sequential_module.named_children()) {
	grand_children[s.name] = s.value;
    }
    const std::string& linear_grandchild_name = node->inputs()[0]->node()->s(attr::name);
    auto grandchild_graph = grand_children[linear_grandchild_name].get_method("forward").graph();
    LinearParameter param;

    auto children_output = grandchild_graph->outputs()[0]->type()->cast<TensorType>();
    auto sizes = children_output->sizes().concrete_sizes().value();
    param.out_features_x  = sizes[0];
    param.out_features_y  = sizes[1];

    auto s = shape(grandchild_graph->inputs()[1]);
    param.in_features_x = s[0];
    param.in_features_y = s[1];

    return param;
  }

  LinearParameter parseLinear1(torch::jit::Node* node) {
    TORCH_CHECK(
        is_module(node, "__torch__.torch.nn.modules.linear.Linear"),
        "node to be Linear");

    LinearParameter param;

    auto value = node->inputs()[1];
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
 
    auto size = pt->sizes().concrete_sizes();
    if (size.has_value()) {
        auto sizes = pt->sizes().concrete_sizes().value();
        param.in_features_x  = sizes[0];
        param.in_features_y  = sizes[1];
    } else {
        TORCH_CHECK( node->kind() == prim::CallMethod,
                     "Kind of node to be prim::CallMethod, but got ",
      	       string(node->kind().toUnqualString()));
        const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
        auto child_graph = children[child_name].get_method("forward").graph();
        auto children_output = child_graph->outputs()[0]->type()->cast<TensorType>();
        auto sizes = children_output->sizes().concrete_sizes().value();
        param.in_features_x  = sizes[0];
        param.in_features_y  = sizes[1];
    }
    param.out_features_x = shape(node->output())[0];
    param.out_features_y = shape(node->output())[1];

    return param;
  }

  void sequential_node(torch::jit::Value* value) {
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto sizes = pt->sizes().concrete_sizes();

    auto node = value->node();
    TORCH_CHECK(
        node->kind() == prim::CallMethod,
        "Kind of node to be prim::CallMethod, but got ",
        string(node->kind().toUnqualString()));

    const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
    auto child_graph = children[child_name].get_method("forward").graph();

    auto nodes = child_graph->nodes();
    for (auto&& node : nodes) {
      node_backend(node);
    }
  }

  std::vector<int64_t> shape(torch::jit::Value* value) {
    auto pt = value->type()->cast<TensorType>();
    TORCH_CHECK(pt);
    auto sizes = pt->sizes().concrete_sizes();

    if (sizes.has_value()) {
      return sizes.value();
    } else {
      auto node = value->node();
      TORCH_CHECK(
          node->kind() == prim::CallMethod,
          "Kind of node to be prim::CallMethod, but got ",
          string(node->kind().toUnqualString()));

      const std::string& child_name = node->inputs()[0]->node()->s(attr::name);
      auto child_graph = children[child_name].get_method("forward").graph();

      auto children_output =
          child_graph->outputs()[0]->type()->cast<TensorType>();
      return children_output->sizes().concrete_sizes().value();
    }
  }

  void allocateValue(torch::jit::Value* value) {
    auto node = value->node();
    auto s = shape(value);
    int64_t n = 0;
    if (s.size() == 0)
      n = 0;
    else
      n = s[0];

    for (int i = 1; i < s.size(); i++) {
      n *= s[i];
    }

    address[value] = allocator->dram_allocate(n);
  }

  void allocateNode(torch::jit::Node* node) {
    auto kind = node->kind();
    if (kind == prim::GetAttr)
      return;
    else if (kind == prim::Constant || kind == prim::ListConstruct)
      return;
    else if (kind == aten::Int || kind == aten::size)
      return;
    else if ((kind != aten::zeros) && (kind != aten::relu) && (kind != prim::NumToTensor) && (kind != aten::cat) && (kind != aten::view) && (is_module(node, "__torch__.torch.nn.modules.container.Sequential"))) {
      return;
    }

    auto outputs = node->outputs();
    for (auto&& i : outputs) {
      allocateValue(i);
    }
  }

  uint64_t allocateConv2dWeight(
      torch::jit::Value* value,
      Conv2dParameter param) {
    auto weight_address = allocator->dram_allocate(
        param.kernel_size_x * param.kernel_size_y * param.in_channels *
        param.out_channels);
    address[value] = weight_address;
    return weight_address;
  }

  void allocateActivationAndInput() {
    auto graph = module.get_method("forward").graph();

    allocateValue(graph->inputs()[1]);
    auto nodes = graph->nodes();
    int i = 0;
    for (auto&& node : nodes) {
      i++;
      allocateNode(node);
    }
  }

  void printAddress() {
    for (const auto& n : address) {
      std::cout << "Value:[" << n.first->debugName() << "] Address:["
                << n.second << "]\n";
    }
  }

  uint64_t ceil_div(uint64_t numerator, uint64_t denominator) {
    auto res = lldiv(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
  }

  Workload get_chiplet_workload(
      Workload total_workload,
      uint64_t Yp,
      uint64_t Kp) {
    assert(Kp != 0);
    assert(Yp != 0);
    return Workload{ceil_div(total_workload.C, Kp),
                    ceil_div(total_workload.H, Yp),
                    total_workload.W};
  }

  Workload get_chiplet_sub_workload(
      Workload chiplet_workload,
      uint64_t Y2,
      uint64_t X2,
      uint64_t K2) {
    assert(K2 != 0);
    assert(Y2 != 0);
    assert(X2 != 0);
    return Workload{ceil_div(chiplet_workload.C, K2),
                    ceil_div(chiplet_workload.H, Y2),
                    ceil_div(chiplet_workload.W, X2)};
  }

  Point get_chiplet_out(
      Workload chiplet_sub_workload,
      uint64_t y2,
      uint64_t x2,
      uint64_t k2) {
    return Point{chiplet_sub_workload.C * k2,
                 chiplet_sub_workload.H * y2,
                 chiplet_sub_workload.W * x2};
  }

  Point chiplet_out_to_total_out(
      Workload chiplet_workload,
      uint64_t kp,
      uint64_t yp,
      Point point_out) {
    return Point{chiplet_workload.C * kp + point_out.C,
                 chiplet_workload.H * yp + point_out.Y,
                 point_out.X};
  }

  Point out_to_in(Point point_out, uint64_t stride_x, uint64_t stride_y) {
    return Point{0, point_out.Y * stride_y, point_out.X * stride_x};
  }
  
  void node_backend(torch::jit::Node*& node) {
    auto kind = node->kind();
    if (kind == prim::GetAttr) {
      return;
    }

    if (kind == aten::relu) {
      char layer_bf[SIZE*2] = "";
      if (layer_num-1) {
        sprintf(layer_bf, " form layer_num:%d type:%s", layer_num_bf, layer_type_bf);
      }
      if (is_module(all_node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
          is_module(all_node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
        param_conv = parseConv2d(all_node_back);
      }

      if (!strncmp(layer_type, "conv", strlen("conv"))) {
        auto total_workload_out_shape = shape(conv_node_back->output());
        auto total_workload_out = Workload{total_workload_out_shape[1],
                                           total_workload_out_shape[2],
                                           total_workload_out_shape[3]};

        param_conv.feature_map_size_x = total_workload_out.H;
        param_conv.feature_map_size_y = total_workload_out.W;
        param_conv.weight = 3;//Conv2D 的权重是通过上位机给出的，这部分和量化关系紧密

        std::cout << "layer_num:" << layer_num << " layer type:" << "conv" << num[0] << layer_bf << "\n";
        std::cout << "Conv param:\nin_channels=" << param_conv.in_channels
                  << " out_channels=" << param_conv.out_channels << " kernel_size_x="
		  << param_conv.kernel_size_x << " kernel_size_y=" << param_conv.kernel_size_y
		  << " stride_x=" << param_conv.stride_x << " stride_y=" << param_conv.stride_y
		  << " dilation_x="<< param_conv.dilation_x << " dilation_y="
		  << param_conv.dilation_y << " transposed=" << param_conv.transposed 
		  << " weight=" << param_conv.weight << " feature_map_size_x=" << param_conv.feature_map_size_x 
		  << " feature_map_size_y=" << param_conv.feature_map_size_y << std::endl;

        auto total_workload_in_shape = shape(conv_node_back->inputs()[1]);
        auto total_workload_in = Workload{total_workload_in_shape[1],
                                          total_workload_in_shape[2],
                                          total_workload_in_shape[3]};

        auto knifeResult = NNKnife();

        auto chiplet_workload_out = get_chiplet_workload(
            total_workload_out, knifeResult.Yp, knifeResult.Kp);
        auto chiplet_sub_workload_out = get_chiplet_sub_workload(
            chiplet_workload_out,
            knifeResult.Y2,
            knifeResult.X2,
            knifeResult.K2);
        auto weight_address = allocateConv2dWeight(GetAttrValue, param_conv);

        for (uint64_t kp = 0; kp < knifeResult.Kp; kp++) {
          for (uint64_t yp = 0; yp < knifeResult.Yp; yp++) {
            uint64_t Chiplet_num = kp * knifeResult.Kp + yp;
            for (uint64_t y2 = 0; y2 < knifeResult.Y2; y2++) {
              for (uint64_t x2 = 0; x2 < knifeResult.X2; x2++) {
                for (uint64_t k2 = 0; k2 < knifeResult.K2; k2++) {
                  auto chiplet_out =
                      get_chiplet_out(chiplet_sub_workload_out, y2, x2, k2);
                  auto total_out = chiplet_out_to_total_out(
                      chiplet_workload_out, kp, yp, chiplet_out);
                  auto total_in =
                      out_to_in(total_out, param_conv.stride_x, param_conv.stride_y);
                  uint64_t act_addr;
                  if ("input" == conv_node_back->inputs()[1]->debugName()) {
                    act_addr = input_to_address(
                        total_workload_in, total_in.C, total_in.Y, total_in.X, address[conv_node_back->inputs()[1]]);
                  } else {
                    act_addr = activition_to_address(
                        total_workload_in,
                        knifeResult.Kp,
                        total_in.C,
                        total_in.Y,
                        total_in.X, address[conv_node_back->inputs()[1]]);
                  }
                }
              }
            }
          }
        }

        if (param_bn.flag) {
	  std::cout << "BN param: dimension " << param_bn.dimen << std::endl;
	  bzero(&param_bn, sizeof(BatchNormParameter));
	}
       layer_num_bf = layer_num;
       sprintf(layer_type_bf, "conv%d", num[0]);
      }

      std::cout << "relu_param: 0_32" << " relu_en 1" << " relu_mode 00" << std::endl;

      return;
    }

    if (kind == aten::leaky_relu) {
      std::cout << "relu_en 1" << std::endl;
      std::cout << "relu_mode 10" << std::endl;
      std::cout << "relu_param 0.01_32" << std::endl;

      return;
    }

    if (kind == aten::tanh) {
      std::cout << "relu_en 1" << std::endl;
      std::cout << "relu_mode 11" << std::endl;
      std::cout << "relu_param 0_32" << std::endl;

      return;
    }

    if (kind == prim::CallMethod) {
      GetAttrValue = node->inputs()[0];
      if (is_module(node, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(node, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
	all_node_back  = node;
	conv_node_back = node;
	num[0]++;
	if (num[0] == 1) {
	  sprintf(layer_type_bf, "conv");
	}
        layer_num++;
        sprintf(layer_type, "conv%d", num[0]);

	return;
      }

      else if (is_module(node, "__torch__.torch.nn.modules.batchnorm.BatchNorm2d")) {
	if (!strncmp(layer_type_bf, "conv", strlen("conv")) || 
	    !strncmp(layer_type_bf, "pool", strlen("pool"))) {
	  param_bn   = parseBatchNorm(node);
          param_conv = parseConv2d(all_node_back);
	  all_node_back   = node;
	  return;
	}
	else {
      	 std::cout << "before BatchNorm layer is wrong..." << std::endl;
	 return;
 	}
      }

      else if (is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d") ||
          is_module(node, "__torch__.torch.nn.modules.pooling.AvgPool2d")) {
        if (is_module(all_node_back, "__torch__.torch.nn.modules.conv.Conv2d") ||
            is_module(all_node_back, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
	  layer_num_bf = layer_num;
	  memcpy(layer_type_bf, layer_type, sizeof(layer_type));
      }
	num[1]++;
        layer_num++;
	all_node_back  = node;
        sprintf(layer_type, "pool%d", num[1]);
        char layer_bf[SIZE*2] = "";
        if (layer_num-1) {
          sprintf(layer_bf, " form layer_num:%d type:%s", layer_num_bf, layer_type_bf);
        }
	std::cout << "layer_num:" << layer_num << "\nlayer type:" << layer_type  << layer_bf << "\n";
        std::cout << "Pooling_en 1" << std::endl;
        auto param = parsePool2d(node);
        auto size = param.kernel_size_x * param.kernel_size_y;
        std::cout << "Pool param: pool_size " << size - 1 << " kernel_size_x " << param.kernel_size_x << 
                  " kernel_size_y " << param.kernel_size_y << " Pooling_en 1" << " oprands " << 1.0 / size 
                  << " stride_x " << param.stride_x << " stride_y " << param.stride_y << std::endl;
        layer_num_bf = layer_num;
        sprintf(layer_type_bf, "pool%d", num[1]);

        return;
      }

      else if (is_module(node, "__torch__.quant_layer.QuantLayer")) {
          return;
     }

      else if (is_module(node, "__torch__.torch.nn.modules.dropout.Dropout")) {
          return;
     }

      else if (is_module(node, "__torch__.torch.nn.modules.container.Sequential")) {
          sequential_node(node->outputs()[0]);
          return;
     }

      else if (is_module(node, "__torch__.torch.nn.modules.activation.ReLU")) {
          return;
     }

      else if (is_module(node, "__torch__.torch.nn.modules.linear.Linear")) {
          num[2]++;
          layer_num++;
	  all_node_back  = node;
          sprintf(layer_type, "fc%d", num[2]);
          char layer_bf[SIZE*2] = "";
          if (layer_num-1) {
            sprintf(layer_bf, " form layer_num:%d type:%s", layer_num_bf, layer_type_bf);
          }
	  std::cout << "layer_num:" << layer_num << "\nlayer type:" << layer_type  << layer_bf << "\n";
          auto param = parseLinear(node);
	  std::cout << "fc param:" << "in_features_x:" << param.in_features_x << " in_features_y:" 
                    << param.in_features_y << " out_features_x:" << param.out_features_x 
                    << " out_features_y:" << param.out_features_y << std::endl;
          layer_num_bf = layer_num;
          sprintf(layer_type_bf, "fc%d", num[2]);
          return;
      }

      auto type = GetAttrValue->type()->cast<c10::ClassType>();
      TORCH_CHECK(type && type->name());
      std::cout << type->name()->qualifiedName() << std::endl;

      TORCH_CHECK(false);
      return;
    }
  }
	  
  void node_one_bkend(torch::jit::Node*& node){
	auto kind = node->kind();
	if (kind == prim::GetAttr) {
	  return;
	}

	if (kind == aten::relu) {
	  std::cout << "relu_en 1" << std::endl;
	  std::cout << "relu_mode 00" << std::endl;
	  std::cout << "relu_param 0_32" << std::endl;

	  return;
	}

	if (kind == aten::leaky_relu) {
	  std::cout << "relu_en 1" << std::endl;
	  std::cout << "relu_mode 10" << std::endl;
	  std::cout << "relu_param 0.01_32" << std::endl;

	  return;
	}

	if (kind == aten::tanh) {
	  std::cout << "relu_en 1" << std::endl;
	  std::cout << "relu_mode 11" << std::endl;
	  std::cout << "relu_param 0_32" << std::endl;

	  return;
	}

	if (kind == prim::CallMethod) {
	  auto GetAttrValue = node->inputs()[0];
	  if (is_module(node, "__torch__.torch.nn.modules.conv.Conv2d") ||
			is_module(node, "__torch__.torch.nn.modules.conv.ConvTranspose2d")) {
		auto param = parseConv2d(node);

		auto total_workload_out_shape = shape(node->output());
		auto total_workload_out = Workload{total_workload_out_shape[1],
						   total_workload_out_shape[2],
					           total_workload_out_shape[3]};

		auto total_workload_in_shape = shape(node->inputs()[1]);
		auto total_workload_in = Workload{total_workload_in_shape[1],
					          total_workload_in_shape[2],
					          total_workload_in_shape[3]};

		auto knifeResult = NNKnife();

		auto chiplet_workload_out = get_chiplet_workload(
			total_workload_out, knifeResult.Yp, knifeResult.Kp);
		auto chiplet_sub_workload_out = get_chiplet_sub_workload(
			chiplet_workload_out,
			knifeResult.Y2,
			knifeResult.X2,
			knifeResult.K2);
		auto weight_address = allocateConv2dWeight(GetAttrValue, param);

		for (uint64_t kp = 0; kp < knifeResult.Kp; kp++) {
		  for (uint64_t yp = 0; yp < knifeResult.Yp; yp++) {
			uint64_t Chiplet_num = kp * knifeResult.Kp + yp;
			for (uint64_t y2 = 0; y2 < knifeResult.Y2; y2++) {
			  for (uint64_t x2 = 0; x2 < knifeResult.X2; x2++) {
				for (uint64_t k2 = 0; k2 < knifeResult.K2; k2++) {
				  auto chiplet_out =
					  get_chiplet_out(chiplet_sub_workload_out, y2, x2, k2);
				  auto total_out = chiplet_out_to_total_out(
					  chiplet_workload_out, kp, yp, chiplet_out);
				  auto total_in =
					  out_to_in(total_out, param.stride_x, param.stride_y);
				  uint64_t act_addr;
				  if ("input" == node->inputs()[1]->debugName()) {
					act_addr = input_to_address(
						total_workload_in, total_in.C, total_in.Y, total_in.X, address[node->inputs()[1]]);
				  } else {
					act_addr = activition_to_address(
						total_workload_in,
						knifeResult.Kp,
						total_in.C,
						total_in.Y,
						total_in.X, address[node->inputs()[1]]);
				  }
			      }
			  }
		      }
		  }
		}

		return;
	  }

      	  else if (is_module(node, "__torch__.torch.nn.modules.pooling.MaxPool2d") ||
              is_module(node, "__torch__.torch.nn.modules.pooling.AvgPool2d")) {
		std::cout << "Pooling_en 1" << std::endl;
		auto param = parsePool2d(node);
		auto size = param.kernel_size_x * param.kernel_size_y;
		std::cout << "pool_size " << size - 1 << std::endl;
		std::cout << "oprands " << 1.0 / size << std::endl;
		return;
	  }

          else if (is_module(node, "__torch__.quant_layer.QuantLayer")) {
              return;
         }

          else if (is_module(node, "__torch__.torch.nn.modules.dropout.Dropout")) {
              return;
         }

          else if (is_module(node, "__torch__.torch.nn.modules.container.Sequential")) {
              return;
          }

          else if (is_module(node, "__torch__.torch.nn.modules.activation.ReLU")) {
              return;
         }

          else if (is_module(node, "__torch__.torch.nn.modules.linear.Linear")) {
              num[2]++;
              auto param = parseLinear1(node);
	      std::cout << "fc param:" << "in_features_x:" << param.in_features_x << " in_features_y:" 
                        << param.in_features_y << " out_features_x:" << param.out_features_x 
                        << " out_features_y:" << param.out_features_y << std::endl;
              return;
          }
	  
	  auto type = GetAttrValue->type()->cast<c10::ClassType>();
	  TORCH_CHECK(type && type->name());
	  std::cout << type->name()->qualifiedName() << std::endl;

	  TORCH_CHECK(false);
	  return;
	}

  }

  void backend() {
    auto nodes = module.get_method("forward").graph()->nodes();
    auto node_num = 0;
    for (auto&& node : nodes) {
      node_num++;
    }
    for (auto&& node : nodes) {
      if (node_num == 2) {
	node_one_bkend(node);
	break;
      } else {
        node_backend(node);
      }
    }
  }
};
} // namespace fun
