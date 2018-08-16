#if !(defined _WIN32)

#include "torch/csrc/jit/fusers/cpu/cpu_fusion_function.h"

#include "torch/csrc/jit/code_template.h"
#include "torch/csrc/jit/resource_guard.h"
#include "torch/csrc/jit/constants.h"

#include "torch/csrc/variable_tensor_functions.h"

#include "ATen/DeviceGuard.h"

namespace torch { namespace jit { namespace cpufuser {

// static const std::string so_template = "/tmp/pytorch_fuserXXXXXX.so";
// static const std::string cpp_template = "/tmp/pytorch_fuserXXXXXX.cpp";

// // NB: -march=native not supported on PPC64 g++.  It's a bit annoying
// // to do a configure-style test to decide whether or not the g++
// // actually supports it or not, so we heuristically use the host
// // compiler to predict if the runtime compiler supports the option we
// // want.  This probably won't work if you're cross-compiling.
// // NB: -march=native is disabled because it has caused problems where
// // compiler and assembler do not agree on what native instruction they
// // understand for AVX512. When we need better CPU performance this
// // optimization can be re-enabled by tracking down the platforms where
// // this error occurs and only selectively disabling it.
// static const std::string compile_string =
//   "\"${cxx}\" -O3 -g "
// #ifndef __PPC64__
// //  "-march=native "
// #endif
//   "-std=c++11 -fPIC ${fopenmp} -shared \"${cpp_file}\" -o \"${so_file}\" -lm";

// static void runCompiler(
//   CPUFusionCompilerConfig& config
// , const std::string& cpp_file
// , const std::string& so_file) {
//   TemplateEnv env;
//   env.s("cxx", config.cxx);
//   env.s("fopenmp", config.openmp ? "-fopenmp" : "");
//   env.s("cpp_file",cpp_file);
//   env.s("so_file",so_file);
//   std::string result = format(compile_string,env);
//   int r = system(result.c_str());
//   if(config.openmp && r != 0) {
//     std::cerr << "warning: pytorch jit fuser failed to compile with openmp, trying without it...\n";
//     config.openmp = false; // disable for future compiles
//     return runCompiler(config, cpp_file, so_file);
//   }
//   JIT_ASSERTM(r == 0, "Failed to compile a fused CPU kernel");
// }

//   // Tries to compress sizes and strides according to cont. Emits the result t
// // c_sizes, c_strides and throws an error on failure (if can't compress)
// static void compressContiguous(
//   at::IntList sizes
// , at::IntList strides
// , const std::vector<bool>& cont
// , uint32_t* c_sizes
// , uint32_t* c_strides) {
//   size_t compressed_dims = 0;
//   size_t cur = 0;
//   size_t ndim = sizes.size();
//   while (cur < ndim) {
//     size_t total_size = sizes[cur];
//     cur++;
//     while(cont[cur-1] && cur < ndim) {
//       JIT_ASSERT(strides[cur-1] == sizes[cur]*strides[cur]);
//       total_size *= sizes[cur];
//       cur++;
//     }
//    // cur starts pointing at the beginning of run to compress
//    // cur ends one _after_ the terminating false or end of list.
//    // total_size is the size of all dimensions [begin,end)
//    // examples:
//    // f = not cont.
//    // t = cont.
//    // x = don't care, including past end of list
//    // s = start of cur
//    // e = end of cur


//    // f x x x
//    // s e

//    //  t f x x
//    //  s   e

//    //  t t f x
//    //  s     e

//     c_sizes[compressed_dims] = total_size;
//     c_strides[compressed_dims] = strides[cur-1];
//     compressed_dims++;
//   }
//   JIT_ASSERT(!cont.back() || strides.back() == 1);
// }

// /*with type_as not checking type of its input, a fusion group can have non-fp32 tensor as input.
// Correct code for this case is generated, however, nvrtc does not know how to handle int*_t integer types,
// so typedefs help it handle those cases*/

// auto type_declarations_template = CodeTemplate(R"(
// typedef ${IndexType} IndexType;
// template<typename T, size_t N>
// struct TensorInfo {
//   T * data;
//   IndexType sizes[N];
//   IndexType strides[N];
// };
// )");

// auto cpu_compilation_unit_template = CodeTemplate(R"(
// #include <cstddef>
// #include <cstdint>
// #include <math.h>
// ${type_declarations}
// #define OMP_THRESHOLD 100000
// static void ${kernelName}_kernel(IndexType totalElements, ${formals}) {
//   #pragma omp parallel for if(totalElements > OMP_THRESHOLD)
//   for (IndexType linearIndex = 0;
//         linearIndex < totalElements;
//         linearIndex += 1) {
//       // Convert `linearIndex` into an offset of tensor:
//       ${tensorOffsets}
//       // calculate the results
//       ${kernelBody}
//     }
// }
// extern "C"
// void ${kernelName}(IndexType totalElements, void ** args) {
//   ${kernelName}_kernel(totalElements ${,argument_loads});
// }
// )");

// // curDimIndex = linearId % sizes[i]; // % sizes[i] is not needed for d == 0, because we already guard for numel outside the index calculation
// // offset += curDimIndex*strides[i]; // *strides[i] is optional if list_is_cont becaause strides.back() == 1
// // linearId /= sizes[i];
// auto dim_calc = CodeTemplate(R"(
// //printf("tensor ${tensor} sizes[${d}] = %d, strides[${d}] = %d\n", ${tensor}.sizes[${d}],${tensor}.strides[${d}]);
// size_t ${tensor}_dimIndex${d} = ${tensor}_linearIndex ${mod_sizes};
// ${tensor}_offset += ${tensor}_dimIndex${d} ${times_stride};
// )");

// static void emitIndexingFor(
//   std::ostream& out
// , const std::string& tensor
// , int ndim
// , bool last_is_cont) {
//   TemplateEnv env;
//   env.s("tensor",tensor);
//   out << format("IndexType ${tensor}_offset = 0;\n",env);
//   out << format("IndexType ${tensor}_linearIndex = linearIndex;\n",env);
//   for(int d = ndim - 1; d >= 0; --d) {
//     env.d("d",d);
//     env.s("mod_sizes", d > 0 ? format("% ${tensor}.sizes[${d}]",env) : "");
//     env.s("times_stride",(d < ndim - 1 || !last_is_cont) ?
//       format("* ${tensor}.strides[${d}]",env) : "");
//     out << dim_calc.format(env);
//     if(d > 0) {
//       out << format("${tensor}_linearIndex /= ${tensor}.sizes[${d}];\n",env);
//     }
//   }
// }

// static std::string valueName(Value* n) {
//   return "n" + std::to_string(n->unique());
// }

// static std::string scalarValue(int64_t v) {
//   return std::to_string(v);
// }

// static std::string scalarValue(double v) {
//   std::ostringstream out;
//   out << std::scientific << v << "f";
//   return out.str();
// }

// static const char* scalarTypeName(at::ScalarType type) {
//   if (type == at::ScalarType::Half) {
//     return "half";
//   }

//   switch(type) {
//     #define DEFINE_CASE(ctype,name,_) \
//       case at::ScalarType::name: return #ctype;
//     AT_FORALL_SCALAR_TYPES_EXCEPT_HALF(DEFINE_CASE)
//     #undef DEFINE_CASE
//     default:
//       throw std::runtime_error("unknown scalar type");
//   }
// }

// static std::string encodeRHS(Node* n) {
//   static std::unordered_map<NodeKind, std::string> simple_map_ops = {
//     // unary
//     {aten::abs, "absf(${0})"},
//     {aten::sigmoid, "1.f / (1.f + expf(-${0}))"},
//     {aten::relu, "${0} < 0 ? 0.f : ${0} "},
//     {aten::log, "logf(${0})"},
//     {aten::log10, "log10f(${0})"},
//     {aten::log1p, "log1pf(${0})"},
//     {aten::log2,  "log2f(${0})"},
//     {aten::lgamma, "lgammaf(${0})"},
//     {aten::exp, "expf(${0})"},
//     {aten::expm1, "expm1f(${0})"},
//     {aten::cos, "cosf(${0})"},
//     {aten::acos, "acosf(${0})"},
//     {aten::cosh, "coshf(${0})"},
//     {aten::sin, "sinf(${0})"},
//     {aten::asin, "asinf(${0})"},
//     {aten::sinh, "sinhf(${0})"},
//     {aten::tan, "tanf(${0})"},
//     {aten::atan, "atanf(${0})"},
//     {aten::tanh, "tanhf(${0})"},
//     {aten::sqrt, "sqrtf(${0})"},
//     {aten::rsqrt, "rsqrtf(${0})"},
//     {aten::ceil, "ceilf(${0})"},
//     {aten::floor, "floorf(${0})"},
//     {aten::round, "roundf(${0})"},
//     {aten::trunc, "truncf(${0})"},
//     {aten::frac, "fracf(${0})"},
//     {aten::reciprocal, "reciprocalf(${0})"},
//     {aten::neg, "-${0}"},
//     //simple binary
//     {aten::atan2, "atan2(${0}, ${1})"},
//     {aten::min, "fminf(${0}, ${1})"},
//     {aten::max, "fmaxf(${0}, ${1})"},

//     //binary with other
//     // TODO: some of these ops will not get generated because
//     // we only work on float inputs/outputs, but they are here to record
//     // that they are valid mappable ops once we handle more type
//     {aten::__and__, "${0} && ${1}"},
//     {aten::__lshift__, "${0} << ${1}"},
//     {aten::__or__, "${0} || ${1}"},
//     {aten::__rshift__, "${0} >> ${1}"},
//     {aten::__xor__, "${0} ^ ${1}"},
//     {aten::div, "${0} / ${1}"},
//     {aten::eq, "${0} == ${1}"},
//     {aten::fmod, "fmodf(${0}, ${1})"},
//     {aten::ge, "(${0} >= ${1})"},
//     {aten::gt, "${0} > ${1}"},
//     {aten::le, "(${0} <= ${1})"},
//     {aten::lt, "${0} < ${1}"},
//     {aten::type_as, "(${0})"}, //everything is implicitly convertible to float
//     {aten::mul, "${0} * ${1}"},
//     {aten::ne, "${0} != ${1}"},
//     {aten::remainder, "remainderf(${0}, ${1})"},
//     {aten::pow, "powf(${0}, ${1})"},

//     //alpha
//     {aten::add, "${0} + ${2}*${1}"},
//     {aten::sub, "(${0} - ${2}*${1})"},
//     {aten::rand_like, "uniform(rnd())"},

//     // simple derivatives
//     {aten::_sigmoid_backward, "${0} * ${1} * (1.f - ${1})"},
//     {aten::_tanh_backward,    "${0} * (1.f - ${1} * ${1})"},
//   };

//   if (n->kind() == prim::Constant) {
//     auto val = toIValue(n->output()).value();
//     if (val.isDouble()) {
//       return scalarValue(val.toDouble());
//     } else {
//       JIT_ASSERT(val.isInt());
//       return scalarValue(val.toInt());
//     }
//   }

//   TemplateEnv env;
//   size_t i = 0;
//   for(auto in : n->inputs()) {
//     env.s(std::to_string(i++), valueName(in));
//   }

//   const auto& str = simple_map_ops.at(n->kind());
//   return format(str, env);
// }

// static std::pair<std::vector<ConcatDesc>, bool> emitCompilationUnit(
//     std::ostream& out,
//     const std::string& name,
//     AnnotatedGraph& agraph,
//     bool use_cuda) {
//   bool has_random = false;
//   Graph& subgraph = *agraph.graph;
//   TemplateEnv env;
//   env.s("kernelName",name);
//   // TODO: handle cases where we need to generate > 2^32 element tensors
//   env.s("IndexType","unsigned int"); //avoiding slow header includes to get uint32_t

//   std::stringstream body;
//   std::stringstream tensorOffsets;
//   std::vector<std::string> formals;
//   std::vector<std::string> argument_loads;
//   auto emitFormal = [&](Value* n, const TensorDesc& desc) {
//     std::string tensor = "t" + std::to_string(formals.size()); //can't be unique() because Param may be an output
//     size_t nDim = desc.nDim();
//     emitIndexingFor(tensorOffsets, tensor, nDim,  desc.lastIsContiguous());
//     env.s("tensor",tensor);
//     env.d("formal_index", formals.size() + 1); // + 1 because the first argument is the linearIndex
//     env.d("nDim",nDim);
//     env.s("scalar_type",scalarTypeName(desc.scalar_type));
//     formals.push_back(format("TensorInfo<${scalar_type},${nDim}> ${tensor}",env));
//     argument_loads.push_back(format("*static_cast<TensorInfo<${scalar_type},${nDim}>*>(args[${formal_index}])",env));
//   };
//   {
//     size_t i = 0;
//     for(auto p : subgraph.inputs())
//       emitFormal(p,agraph.input_desc[i++]);
//   }
//   std::vector<ConcatDesc> concat_desc;
//   std::vector<Value*> flat_output_nodes;
//   {
//     size_t i = 0;
//     for(auto o : subgraph.outputs()) {
//       auto & desc = agraph.output_desc[i++];
//       if(o->node()->kind() != prim::FusedConcat) {
//         emitFormal(o, desc);
//         concat_desc.emplace_back();
//         flat_output_nodes.push_back(o);
//       } else {
//         auto cat = o->node();
//         concat_desc.emplace_back(desc, cat->inputs().size(), cat->i(attr::dim));
//         for(auto c : cat->inputs()) {
//           emitFormal(c, *concat_desc.back().subtensorDesc);
//           flat_output_nodes.push_back(c);
//         }
//       }
//     }
//   }

//   size_t formal_count = 0;
//   for(auto p : subgraph.inputs()) {
//     env.s("node",valueName(p));
//     env.d("formal",formal_count++);

//     // Acquires and converts (if needed) inputs
//     auto pt = p->type()->cast<TensorType>();
//     env.s("access", format("t${formal}.data[t${formal}_offset]", env));

//     //TODO: actual type propagation rather than relying on auto..
//     body << format("auto ${node} = ${access};\n",env);
//   }

//   for(auto n : subgraph.nodes()) {
//     // FusedConcat nodes work by narrowing the output Tensors before the kernel runs
//     if (n->kind() == prim::FusedConcat)
//       continue;
//     if(n->kind() == aten::rand_like) {
//       has_random = true;
//       if(!use_cuda)
//         throw std::runtime_error("Fusion doesn't support rand on CPU");
//     }
//     env.s("node",valueName(n->output()));
//     env.s("rhs", encodeRHS(n));
//     body << format("auto ${node} = ${rhs};\n",env);
//   }

//   for(auto o : flat_output_nodes) {
//     env.d("formal",formal_count++);
//     env.s("access",format("t${formal}.data[t${formal}_offset]",env));
//     env.s("node",valueName(o));

//     // Acquires and converts (if needed) outputs
//     auto ot = o->type()->cast<TensorType>();
//     body << format("${access} = ${node};\n",env);
//   }

//   env.s("tensorOffsets",tensorOffsets.str());
//   env.s("kernelBody",body.str());
//   env.v("formals",formals);
//   env.v("argument_loads",argument_loads);
//   env.s("type_declarations", type_declarations_template.format(env));
//   out << cpu_compilation_unit_template.format(env);

//   return std::make_pair(std::move(concat_desc), has_random);
// }

// // Host-side view of TensorInfo (that visivle for the kernel is defined above).
// // Note dims[0] - we need to dynamically allocate the dims.
// struct TensorInfo {
//   void* data;
//   #pragma GCC diagnostic ignored "-Wpedantic"
//     uint32_t sizes_strides[0];
//   #pragma GCC diagnostic pop

//   uint32_t* sizes(size_t nDim) { return &sizes_strides[0]; }
//   uint32_t* strides(size_t nDim) { return &sizes_strides[nDim]; }
// };

// static const std::string disas_string =
//   "objdump -M  intel -d \"${so_file}\"";

// static void disas(const std::string& so_file) {
//   TemplateEnv env;
//   env.s("so_file", so_file);
//   std::string cmd = format(disas_string, env);
//   int r = system(cmd.c_str());
//   JIT_ASSERT(r == 0);
// }

// CPUFusionFunction::CPUFusionFunction(
//   const std::string& name
// , AnnotatedGraph& agraph
// , CPUFusionCompilerConfig& config)
// : name{name}, input_desc{agraph.input_desc}, output_desc{agraph.output_desc} {
//   TempFile so_file(so_template, 3);
//   TempFile cpp_file(cpp_template, 4);

//   std::stringstream cu;
//   auto ret = emitCompilationUnit(cu, name, agraph, false);
//   concat_desc = std::move(ret.first);
//   has_random = ret.second;
//   JIT_ASSERT(!has_random);
//   compilation_unit = cu.str();
//   cpp_file.write(compilation_unit);
//   cpp_file.sync();
//   runCompiler(config, cpp_file.name(), so_file.name());
//   if(config.debug) disas(so_file.name());
//   so_lib.reset(new DynamicLibrary(so_file.name().c_str()));
//   #pragma GCC diagnostic ignored "-Wpedantic"
//     kernel = reinterpret_cast<void(*)(uint32_t, void**)>(so_lib->sym(name.c_str()));
//   #pragma GCC diagnostic pop
// }

// void CPUFusionFunction::launch_with_tensors(
//   at::ArrayRef<at::Tensor> inputs
// , at::ArrayRef<at::Tensor> outputs) {
//   at::DeviceGuard device_guard(inputs);
//   JIT_ASSERT(inputs.size() == input_desc.size());
//   JIT_ASSERT(outputs.size() == output_desc.size());
//   size_t flat_outputs_size = 0;
//   for(auto& c : concat_desc)
//     flat_outputs_size += c.nSubtensors;
//   // XXX: this code assumes that inputs are 32-bit addressable
//   // XXX: this code assumes that all inputs are of the same size
//   JIT_ASSERT(inputs[0].numel() <= std::numeric_limits<uint32_t>::max());
//   uint32_t numel = inputs[0].numel();
//   at::IntList map_size = inputs[0].sizes();
//   // Compute the storage needed to store TensorInfo structs for inputs and outputs.
//   size_t uncompressedDim = input_desc.at(0).contiguity.size();
//   size_t maxPossibleTensorInfoSize = sizeof(TensorInfo) + 2 * sizeof(uint32_t) * uncompressedDim;
//   size_t maxPossibleBufferSize = maxPossibleTensorInfoSize * (inputs.size() + flat_outputs_size);
//   std::vector<char> buffer(maxPossibleBufferSize);
//   char* buffer_next = buffer.data();
//   // A vector of arguments to the kernel. It's (numel, *input_descs, *output_descs)
//   std::vector<void*> arguments;
//   arguments.reserve(3 + inputs.size() + flat_outputs_size);
//   // Asserts that t's dims can be compressed in the same way as in desc
//   // (that's what the kernel assumes), and appends it to the arguments vector.
//   auto addTensorInfo = [&](TensorDesc& desc, const at::Tensor& t) {
//     size_t nDim = desc.nDim(); // NOTE: this is the compressed dim
//     JIT_ASSERT(nDim <= uncompressedDim); // We'd overflow the space otherwise
//     auto ti = reinterpret_cast<TensorInfo*>(buffer_next);
//     ti->data = t.data_ptr();
//     compressContiguous(t.sizes(), t.strides(), desc.contiguity, ti->sizes(nDim), ti->strides(nDim));
//     buffer_next += maxPossibleTensorInfoSize;
//     arguments.push_back(ti);
//   };

//   arguments.push_back(&numel);
//   for (size_t i = 0; i < input_desc.size(); ++i)
//     addTensorInfo(input_desc[i], inputs[i]);
//   for (size_t i = 0; i < output_desc.size(); ++i) {
//     auto& c = concat_desc[i];
//     at::Tensor o = outputs[i];
//     if(c.nSubtensors == 1) {
//       o.resize_(map_size);
//       addTensorInfo(output_desc[i], outputs[i]);
//     } else {
//       size_t small_size = map_size[c.dim];
//       std::vector<int64_t> concat_size(map_size.begin(), map_size.end());
//       concat_size[c.dim] = small_size * c.nSubtensors;
//       o.resize_(concat_size);
//       size_t offset = 0;
//       for(size_t j = 0; j < c.nSubtensors; ++j) {
//         // because the concatenated_output stays live, the underlying data
//         // in this view remains live through the end of this function
//         // so there is not need to hold onto this tensor
//         auto view = o.narrow(c.dim, offset, small_size);
//         addTensorInfo(*c.subtensorDesc, view);
//         offset += small_size;
//       }
//     }
//   }

//   launch_raw(numel, arguments.data());
// }

// void CPUFusionFunction::launch(
//   at::ArrayRef<at::Tensor> inputs
// , std::vector<at::Tensor>& outputs) {
//   at::DeviceGuard guard(inputs.back());
//   outputs.clear();
//   outputs.reserve(outputDescriptors().size());
//   for (auto& od : outputDescriptors()) {
//     outputs.push_back(torch::getType(backend(),od.scalar_type).tensor());
//   }

//   launch_with_tensors(inputs, outputs);
// }

} // namespace cpufuser
} // namespace jit
} // namespace torch

#endif // !(defined _WIN32)
