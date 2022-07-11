#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/backends/backend_detail.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/serialization/import.h>
#include <torch/torch.h>

// Tests go in torch::jit
namespace torch {
namespace jit {
TEST(BackendTest, ToBackend) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs).toTupleRef().elements().vec();

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "test_backend", m, compile_spec, any_dict_ty);
  // lowered module code:
  /*
    class test_backendLoweredModule(Module):
      __parameters__ = []
      __buffers__ = []
      __processed_module : Any
      __method_compile_spec : Dict[str, Any]
      __backend : __torch__.torch.classes.__backends__.test_backend
      __handles : Dict[str, Any]
      def __create_backend(self: torch.jit.test_backendLoweredModule) -> None:
        _0 =
    __torch__.torch.classes.__backends__.test_backend.__new__(__torch__.torch.classes.__backends__.test_backend)
        _1 = (_0).__init__()
        self.__backend = _0
        return None
      def __getstate__(self: torch.jit.test_backendLoweredModule) ->
    Tuple[Dict[str, Any], Any]: _2 = (self.__method_compile_spec,
    self.__processed_module) return _2 def __setstate__(self:
    torch.jit.test_backendLoweredModule, state: Tuple[Dict[str, Any], Any]) ->
    None: self.__method_compile_spec = (state)[0] self.__processed_module =
    (state)[1] _3 = (self).__create_backend() _4 =
    (self.__backend).compile(self.__processed_module,
    self.__method_compile_spec, ) self.__handles = _4 return None def
    forward(self: torch.jit.test_backendLoweredModule, x: Tensor, h: Tensor) ->
    Tuple[Tensor, Tensor]: _5 = uninitialized(Tensor) typed_inputs =
    annotate(List[Any], [x, h]) _6 =
    (self.__backend).execute((self.__handles)["forward"], typed_inputs, ) _7,
    _8, = _6 _9 = isinstance(_7, Tensor) if _9: _10 = unchecked_cast(Tensor, _7)
        else:
          ops.prim.RaiseException("AssertionError: ")
          _10 = _5
        _11 = isinstance(_8, Tensor)
        if _11:
          _12 = unchecked_cast(Tensor, _8)
        else:
          ops.prim.RaiseException("AssertionError: ")
          _12 = _5
        return (_10, _12)

   */
  auto res = lm.forward(inputs).toTupleRef().elements().vec();
  AT_ASSERT(res[0].toTensor().equal(ref[0].toTensor()));
  AT_ASSERT(res[1].toTensor().equal(ref[1].toTensor()));
}

TEST(BackendTest, ToBackendNotAvailable) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return self.accum(x, h), self.sub_accum(x, h)

    def accum(self, x, h):
        return x + h

    def sub_accum(self, x, h):
        return x - h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs).toTupleRef().elements().vec();

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // Produce lowered module (backend not available).
  // Exception is not thrown at this point.
  auto lm = torch::jit::detail::codegen_backend_module(
      "test_backend_unavailable", m, compile_spec, any_dict_ty);
  // Validate exception is thrown when trying to execute and
  // the backend is not available.
  ASSERT_THROWS_WITH_MESSAGE(
      lm.forward(inputs).toTupleRef().elements(), "Backend is not available.");
}

TEST(BackendTest, TestCompiler) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x + h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs);

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);
  auto res = lm.forward(inputs);
  AT_ASSERT(res.toTensor().equal(ref.toTensor()));

  std::stringstream ss;
  lm._save_for_mobile(ss);
  auto mlm = _load_for_mobile(ss);
  auto mres = mlm.forward(inputs);
  AT_ASSERT(mres.toTensor().equal(ref.toTensor()));
}

TEST(BackendTest, TestCompilerWithStringTable) {
  setShouldUseFormatWithStringTable(true);
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x + h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(2.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto ref = m.forward(inputs);

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);
  auto res = lm.forward(inputs);
  AT_ASSERT(res.toTensor().equal(ref.toTensor()));

  std::stringstream ss;
  lm._save_for_mobile(ss);
  auto mlm = _load_for_mobile(ss);
  auto mres = mlm.forward(inputs);
  setShouldUseFormatWithStringTable(false);
  AT_ASSERT(mres.toTensor().equal(ref.toTensor()));
}

TEST(BackendTest, TestComposite) {
  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

  Module m_add("m_add");
  m_add.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  auto lm_add = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m_add, compile_spec, any_dict_ty);

  Module m_sub("m_sub");
  m_sub.define(R"(
    def forward(self, x, y):
      return x - y
  )");
  auto lm_sub = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m_sub, compile_spec, any_dict_ty);

  Module c("C");
  c.register_module("Add", lm_add);
  c.register_module("Sub", lm_sub);
  c.define(R"(
    def forward(self, x, y):
      return self.Add.forward(x, y) * self.Sub.forward(x, y)
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto res_jit = c.forward(inputs);

  std::stringstream ss;
  c._save_for_mobile(ss);
  auto mc = _load_for_mobile(ss);
  auto res_mobile = mc.forward(inputs);

  AT_ASSERT(res_jit.toTensor().equal(res_mobile.toTensor()));
}

TEST(BackendTest, TestPrimDtype) {
  Module c("name");
  c.define(R"(
    def forward(self, x, y):
      c = y.dtype
      return c
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(1.0 * torch::ones({}));
  auto res_jit = c.forward(inputs);

  std::stringstream ss;
  c._save_for_mobile(ss);
  auto mc = _load_for_mobile(ss);
  auto res_mobile = mc.forward(inputs);

  ASSERT_EQ(res_jit.toInt(), res_mobile.toInt());
}

Module getCompositeModuleWithSameNameSubModules() {
  // Two submodules with same module name but different forward and other
  // functions should be serialized and loaded correctly.

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());

  Module sub1("m_add");
  sub1.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  auto lowered_sub1 = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", sub1, compile_spec, any_dict_ty);

  Module sub2("m_add");
  sub2.define(R"(
    def forward(self, x, y):
      return x - y
  )");
  auto lowered_sub2 = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", sub2, compile_spec, any_dict_ty);

  Module c("C");
  c.register_module("Add", lowered_sub1);
  c.register_module("Sub", lowered_sub2);
  c.define(R"(
    def forward(self, a, b, s:int):
      c = self.Add.forward(a, b)
      d = self.Sub.forward(a, b)
      y = s * (c * d)
      return y
  )");

  return c;
}

TEST(BackendTest, TestCompositeWithSetStates) {
  Module c = getCompositeModuleWithSameNameSubModules();

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::ones({}));
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(3);
  auto res_jit = c.forward(inputs);

  std::stringstream ss;
  c._save_for_mobile(ss);
  auto mc = _load_for_mobile(ss);
  auto res_mobile = mc.forward(inputs);
  AT_ASSERT(res_jit.toTensor().equal(res_mobile.toTensor()));
}

TEST(BackendTest, TestConsistencyOfCompositeWithSetStates) {
  Module c = getCompositeModuleWithSameNameSubModules();

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::ones({}));
  inputs.emplace_back(3.0 * torch::ones({}));
  inputs.emplace_back(3);

  std::stringstream ss, ss_resave;
  c._save_for_mobile(ss);
  auto mc = _load_for_mobile(ss);
  auto res_mobile = mc.forward(inputs);
  ss.seekg(0, ss.beg);

  // check if the methods names are always the same
  // by reloading the script module and saving it back as mobile
  // The below checks ensure that the names of Methods
  // and numerical outputs of mobile and reloaded mobile
  // modules are same.
  auto script_module_load = torch::jit::load(ss);
  script_module_load._save_for_mobile(ss_resave);
  auto mc_reload = _load_for_mobile(ss_resave);
  auto res_mobile_reload = mc_reload.forward(inputs);

  AT_ASSERT(res_mobile_reload.toTensor().equal(res_mobile.toTensor()));

  auto mc_methods = mc.get_methods();
  auto mc_reload_methods = mc_reload.get_methods();

  std::vector<std::string> mc_method_qns, mc_reload_method_qns;

  auto get_qual_name = [](mobile::Method method) -> std::string {
    return method.function().qualname().qualifiedName();
  };

  std::transform(
      mc_methods.begin(),
      mc_methods.end(),
      std::back_inserter(mc_method_qns),
      get_qual_name);

  std::transform(
      mc_reload_methods.begin(),
      mc_reload_methods.end(),
      std::back_inserter(mc_reload_method_qns),
      get_qual_name);

  AT_ASSERT(std::equal(
      mc_method_qns.begin(),
      mc_method_qns.end(),
      mc_reload_method_qns.begin()));
}

TEST(BackendTest, TestCompilerNotSupport) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x * h
  )");

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  ASSERT_THROWS_WITH_MESSAGE(
      torch::jit::detail::codegen_backend_module(
          "backend_with_compiler_demo", m, compile_spec, any_dict_ty),
      "The node of aten::mul is not supported in this compiler. Source code:");
}

TEST(BackendTestDebugInfo, TestCompiler) {
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x + h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);

  std::stringstream ss;
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  auto mlm = _load_for_mobile(ss);
  std::string error_pattern = R"(
  Module hierarchy:top(m)::<unknown>.__loweredModule__(m)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

            def forward(self, x: Tensor, h: Tensor):
                return self.__loweredModule__.forward(x, h)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, h, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, h):
        return x + h
               ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

TEST(BackendTestDebugInfo, TestCompilerWithStringTable) {
  setShouldUseFormatWithStringTable(true);
  Module m("m");
  m.define(R"(
    def forward(self, x, h):
        return x + h
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", m, compile_spec, any_dict_ty);

  std::stringstream ss;
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  auto mlm = _load_for_mobile(ss);
  std::string error_pattern = R"(
  Module hierarchy:top(m)::<unknown>.__loweredModule__(m)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

            def forward(self, x: Tensor, h: Tensor):
                return self.__loweredModule__.forward(x, h)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, h, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, h):
        return x + h
               ~~~~~ <--- HERE
  )";
  setShouldUseFormatWithStringTable(false);
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

TEST(BackendTestDebugInfo, TestExceptionStackForCompilerWithModuleHierarchy) {
  Module a("A");
  a.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  Module b("B");
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", c, compile_spec, any_dict_ty);

  std::stringstream ss;
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  auto mlm = _load_for_mobile(ss);
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.__loweredModule__(C)::forward.A0(A)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

            def forward(self, x: Tensor, y: Tensor):
                return self.__loweredModule__.forward(x, y)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

TEST(
    BackendTestDebugInfo,
    TestExceptionStackForCompilerWithTwoLevelModuleHierarchy) {
  Module a("A");
  a.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  Module b("B");
  b.register_module("A0", a);
  b.define(R"(
    def forward(self, x, y):
      return self.A0.forward(x, y) + 2
  )");
  Module c("C");
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y):
      return self.B0.forward(x, y) + 3
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lm = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", c, compile_spec, any_dict_ty);

  std::stringstream ss;
  lm._save_for_mobile(ss, ExtraFilesMap(), true);
  auto mlm = _load_for_mobile(ss);
  /*
   * Error stack throw will look like this:
   * Module hierarchy:top(backend_with_compiler_demoLoweredModule).B0(B).A0(A)
   * Traceback of TorchScript (most recent call last):
   * File "<string>", line 5, in FunctionName_UNKNOWN
   *               typed_inputs: List[Any] = [x, y, ]
   *               if self.__backend.is_available() :
   *                 _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                       ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                 assert isinstance(_0, Tensor)
   *                 return _0
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.B0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + 2
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   */
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.__loweredModule__(C)::forward.B0(B)::forward.A0(A)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

            def forward(self, x: Tensor, y: Tensor):
                return self.__loweredModule__.forward(x, y)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.B0.forward(x, y) + 3
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

    def forward(self, x, y):
      return self.A0.forward(x, y) + 2
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(mlm.forward(inputs), error_pattern);
}

TEST(BackendTestDebugInfo, TestExceptionStackForCompilerWithLoweredSubModule) {
  std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>();
  Module a("A");
  a.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  Module b("B");
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  IValue submodule = c.attr("A0");
  Module current_sm = submodule.toModule();
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lowered_submodule = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", current_sm, compile_spec, any_dict_ty);

  c.type()->unsafeChangeAttributeType("A0", lowered_submodule.type());
  c.setattr("A0", lowered_submodule._ivalue());
  std::unordered_map<TypePtr, TypePtr> type_remap;
  type_remap[a.type()] = lowered_submodule.type();
  auto type_remap_fn = [&type_remap](TypePtr in) {
    auto it = type_remap.find(in);
    if (it == type_remap.end())
      return in;
    return it->second;
  };
  for (auto& fn : c.type()->methods()) {
    auto method = c.get_method(fn->name());
    auto graph = method.graph();
    graph->remapTypes(type_remap_fn);
    auto new_schema = fn->getSchema().cloneWithRemappedTypes(type_remap_fn);
    fn->setSchema(new_schema);
  }

  std::stringstream ss;
  c._save_for_mobile(ss, ExtraFilesMap(), true);
  auto c_loaded = _load_for_mobile(ss);
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.A0(A)::forward.__loweredModule__(A)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

            def forward(self, x: Tensor, y: Tensor):
                return self.__loweredModule__.forward(x, y)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(c_loaded.forward(inputs), error_pattern);
}

TEST(
    BackendTestDebugInfo,
    TestExceptionStackForCompilerWithSelectiveLoweredSubModule) {
  std::shared_ptr<CompilationUnit> cu = std::make_shared<CompilationUnit>();
  Module aa("AA");
  aa.define(R"(
    def forward(self, x, y):
      return x + y
  )");
  Module a("A");
  a.register_module("AA0", aa);
  a.define(R"(
    def forward(self, x, y):
      return self.AA0.forward(x, y) + 3
  )");
  Module b("B");
  b.define(R"(
    def forward(self, x):
      return x + 2
  )");
  Module c("C");
  c.register_module("A0", a);
  c.register_module("B0", b);
  c.define(R"(
    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
  )");

  std::vector<IValue> inputs;
  inputs.emplace_back(torch::rand({2, 4}));
  inputs.emplace_back(torch::rand({13, 9}));

  c10::Dict<IValue, IValue> compile_spec(StringType::get(), AnyType::get());
  c10::Dict<IValue, IValue> fake_dict(StringType::get(), AnyType::get());
  fake_dict.insert("", "");
  compile_spec.insert("forward", fake_dict);
  IValue submodule = c.attr("A0");
  Module current_sm = submodule.toModule();
  auto any_dict_ty = DictType::create(StringType::get(), AnyType::get());
  // lowered module
  auto lowered_submodule = torch::jit::detail::codegen_backend_module(
      "backend_with_compiler_demo", current_sm, compile_spec, any_dict_ty);

  c.type()->unsafeChangeAttributeType("A0", lowered_submodule.type());
  c.setattr("A0", lowered_submodule._ivalue());
  std::unordered_map<TypePtr, TypePtr> type_remap;
  type_remap[a.type()] = lowered_submodule.type();
  auto type_remap_fn = [&type_remap](TypePtr in) {
    auto it = type_remap.find(in);
    if (it == type_remap.end())
      return in;
    return it->second;
  };
  for (auto& fn : c.type()->methods()) {
    auto method = c.get_method(fn->name());
    auto graph = method.graph();
    graph->remapTypes(type_remap_fn);
    auto new_schema = fn->getSchema().cloneWithRemappedTypes(type_remap_fn);
    fn->setSchema(new_schema);
  }

  std::stringstream ss;
  c._save_for_mobile(ss, ExtraFilesMap(), true);
  auto c_loaded = _load_for_mobile(ss);
  /*
   * Erro stack trace will look like this:
   * Module hierarchy:top(C).A0(backend_with_compiler_demoLoweredModule).AA0(AA)
   * Traceback of TorchScript (most recent call last):
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.A0.forward(x, y) + self.B0.forward(x)
   *             ~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 5, in FunctionName_UNKNOWN
   *                typed_inputs: List[Any] = [x, y, ]
   *                if self.__backend.is_available() :
   *                  _0, = self.__backend.execute(self.__handles["forward"],
   * typed_inputs)
   *                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
   *                  assert isinstance(_0, Tensor)
   *                  return _0
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return self.AA0.forward(x, y) + 3
   *             ~~~~~~~~~~~~~~~~ <--- HERE
   *
   *  File "<string>", line 3, in FunctionName_UNKNOWN
   *
   *    def forward(self, x, y):
   *      return x + y
   *             ~~~~~ <--- HERE
   *
   *
   *  */
  std::string error_pattern = R"(
  Module hierarchy:top(C)::<unknown>.A0(A)::forward.__loweredModule__(A)::forward.AA0(AA)::forward.aten::add
Traceback of TorchScript (most recent call last):
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.A0.forward(x, y) + self.B0.forward(x)
             ~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

            def forward(self, x: Tensor, y: Tensor):
                return self.__loweredModule__.forward(x, y)
                       ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 5, in forward
                typed_inputs: List[Any] = [x, y, ]
                if self.__backend.is_available() :
                  _0, = self.__backend.execute(self.__handles["forward"], typed_inputs)
                        ~~~~~~~~~~~~~~~~~~~~~~ <--- HERE
                  assert isinstance(_0, Tensor)
                  return _0
  File "<string>", line 3, in <unknown>

    def forward(self, x, y):
      return self.AA0.forward(x, y) + 3
             ~~~~~~~~~~~~~~~~ <--- HERE

  File "<string>", line 3, in forward

    def forward(self, x, y):
      return x + y
             ~~~~~ <--- HERE
  )";
  ASSERT_THROWS_WITH_MESSAGE(c_loaded.forward(inputs), error_pattern);
}

} // namespace jit
} // namespace torch
