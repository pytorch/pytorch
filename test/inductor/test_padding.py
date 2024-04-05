# Owner(s): ["module: inductor"]
import copy
import os

import functools
import unittest
import torch
from torch import Tensor
from torch import nn
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import rand_strided
from torch._inductor import config, ir
from torch._inductor.fx_passes import pad_mm as pad_mm_pass
from torch._inductor.utils import do_bench, run_and_get_code
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch._dynamo.utils import maybe_cprofile
from torch._inductor.fx_passes.pad_mm import pad_mm, pad_dim

DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"
DO_ACC_TEST = os.environ.get("DO_ACC_TEST", "1") == "1"

WITH_STACK = os.environ.get("WITH_STACK") == "1"

def get_optim(m):
    return torch.optim.Adam(
        m.parameters(), lr=0.01, capturable=True, foreach=True
    )

def create_timm_model(model_name):
    from timm.models import create_model
    model = create_model(
        model_name,
        in_chans=3,
        scriptable=False,
        num_classes=None,
        drop_rate=0.0,
        drop_path_rate=None,
        drop_block_rate=None,
        # pretrained=True,
    )
    return model

def gen_transformer_inputs(vocab_size, bs, seq_length):
    def geninp():
        return torch.randint(0, vocab_size, (bs, seq_length), dtype=torch.int64, requires_grad=False)

    input_dict = {
        "input_ids": geninp(),
        "labels": geninp()
    }
    return input_dict


class LinearAndSoftmax(nn.Module):
    """
    It's very common that a transformer model will do a matmul and then
    softmax/log_softmax in the end.

    Creating this toy model to capture the pattern and make sure we do
    proper padding.
    """

    def __init__(self, vocab_size=30523, bias=True):
        """
        The default vocab size for BertForMaskedLM is 30522.
        We run a few test cases with good or bad vocab_size around Bert's
        default value.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.linear = nn.Linear(768, vocab_size, bias=bias)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x, label):
        x = self.linear(x)
        return self.ce(x.view(-1, self.vocab_size), label.view(-1))

    def get_example_inputs(self, batch_size=16):
        return torch.randn(batch_size, 512, 768), torch.randint(
            0, self.vocab_size, (batch_size, 512)
        )


def forward_and_backward_pass(m, inputs):
    loss = m(*inputs).sum().backward()


USE_CUDA_GRAPHS = os.environ.get("USE_CUDA_GRAPHS", "1") == "1"
@config.patch(
    {
        "benchmark_kernel": True,
        "triton.unique_kernel_names": True,
        "triton.cudagraphs": USE_CUDA_GRAPHS,
    }
)
class PaddingTest(TestCase):
    def check_close(self, ref, act, tol=1e-3):
        if "LongformerMaskedLMOutput" in str(type(ref)):
            ref = ref.loss
            act = act.loss
        if "SequenceClassifierOutput" in str(type(ref)):
            ref = ref.logits
            act = act.logits
        if isinstance(ref, dict) and "loss" in ref:
            ref = ref["loss"]
            act = act["loss"]
        self.assertTrue(
            torch.allclose(ref, act, atol=tol, rtol=tol), f"ref:\n{ref}\nact:\n{act}"
        )

    def common_numeric_check(self, f, *args, tol=1e-3, **kwargs):
        ref = f(*args, **kwargs)
        opt_f = torch.compile(f)
        act = opt_f(*args, **kwargs)
        self.check_close(ref, act, tol)

    def test_mm_perf(self):
        def naive_mm(a, b):
            return a @ b

        def _compute_padding(s, align):
            return (s + align - 1) // align * align - s

        @torch.compile
        def pad_mm(a, b, align=16):
            """
            NOTE: this function only pad a single dimension which is good
            enough for testing.
            """
            m_padding = _compute_padding(a.size(0), align)
            k_padding = _compute_padding(a.size(1), align)
            n_padding = _compute_padding(b.size(1), align)
            return pad_mm_pass.pad_mm(a, b, m_padding, k_padding, n_padding)

        for M, K, N, f in (
            (8192, 768, 30523, naive_mm),
            (8192, 768, 30523, pad_mm),
            (8192, 768, 30528, naive_mm),
            (30523, 8192, 768, naive_mm),
            (30528, 8192, 768, naive_mm),
        ):
            a = torch.randn(M, K)
            b = torch.randn(K, N)
            ms = do_bench(lambda: f(a, b))
            print(f"MxKxN {M}x{K}x{N} {f.__name__}: {ms:.3f}ms")

    def test_nobias_single(self):
        self.test_single(bias=False)

    def test_nobias_both(self):
        self.test_both(bias=False)

    def test_single(self, bias=True):
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inputs_bad_shape = m_bad_shape.get_example_inputs()
        m_bad_shape_opt = torch.compile(copy.deepcopy(m_bad_shape))

        _, wrapper_codes = run_and_get_code(
            forward_and_backward_pass, m_bad_shape_opt, inputs_bad_shape
        )
        forward_and_backward_pass(m_bad_shape, inputs_bad_shape)
        self.assertTrue(
            torch.allclose(
                m_bad_shape.linear.weight.grad, m_bad_shape_opt.linear.weight.grad
            )
        )
        self.assertTrue(len(wrapper_codes) == 2)  # one for forward and oen for backward
        forward_wrapper = wrapper_codes[0]

        # make sure the store for softmax is aligned
        self.assertTrue(
            "tl.store(out_ptr2 + (r1 + (30528*x0))" in forward_wrapper,
            f"forward_wrapper: {forward_wrapper}",
        )

        if DO_PERF_TEST:
            latency = do_bench(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inputs_bad_shape)
            )
            print(f"latency: {latency:.3f}ms")

    def test_both(self, bias=True):
        m_bad_shape = LinearAndSoftmax(vocab_size=30523, bias=bias)
        inptus_bad_shape = m_bad_shape.get_example_inputs()
        m_good_shape = LinearAndSoftmax(vocab_size=30528, bias=bias)
        inputs_good_shape = m_good_shape.get_example_inputs()

        m_bad_shape_opt = torch.compile(m_bad_shape)
        m_good_shape_opt = torch.compile(m_good_shape)

        if DO_PERF_TEST:
            latency_good_shape = do_bench(
                lambda: forward_and_backward_pass(m_good_shape_opt, inputs_good_shape)
            )
            latency_bad_shape = do_bench(
                lambda: forward_and_backward_pass(m_bad_shape_opt, inptus_bad_shape)
            )
            print(
                f"Latency for good shape v.s. bad shape: {latency_good_shape:.3f}ms v.s. {latency_bad_shape:.3f}ms"
            )

    @config.patch(pattern_matcher=False)
    def test_attention(self):
        batch_size, seq_len, num_heads, hidden_size = 1, 4, 1, 16
        inv_scale = (num_heads / hidden_size) ** 0.5

        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.query = nn.Linear(hidden_size, hidden_size)
                self.key = nn.Linear(hidden_size, hidden_size)
                self.value = nn.Linear(hidden_size, hidden_size)

            @staticmethod
            def reshape(x):
                return x.view(batch_size, seq_len, num_heads, -1).permute(0, 2, 1, 3)

            @staticmethod
            def cancel_reshape(x):
                return x.permute(0, 2, 1, 3).view(batch_size, seq_len, hidden_size)

            def forward(self, x):
                query, key, value = self.query(x), self.key(x), self.value(x)
                weights = (
                    torch.matmul(
                        self.reshape(query), self.reshape(key).permute(0, 1, 3, 2)
                    )
                    * inv_scale
                ).softmax(dim=-1)
                return self.cancel_reshape(torch.matmul(weights, self.reshape(value)))

        attn = Attention()
        x = torch.randn(batch_size, seq_len, hidden_size)

        self.common_numeric_check(attn, x)

    def test_view(self):
        def f(x):
            return x.view(3, 3, 3)

        x = torch.randn(3, 9)
        self.common_numeric_check(f, x)

    def test_pad_strides(self):
        sizes = [2, 16, 511]
        in_strides = [8176, 511, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes))
        expected_strides = [8192, 512, 1]
        self.assertEqual(
            expected_strides, out_strides, f"{expected_strides} v.s. {out_strides}"
        )

    def test_pad_strides_skip(self):
        """
        The padding is skipped to avoid too much memory overhead.
        """
        sizes = [2, 16, 127]
        in_strides = [2032, 127, 1]
        out_strides = list(ir.Layout._pad_strides(in_strides, sizes))
        expected_strides = [2032, 127, 1]
        self.assertEqual(
            expected_strides, out_strides, f"{expected_strides} v.s. {out_strides}"
        )

    def test_pad_3d_tensor(self):
        """
        Constructing this test case guided by the fact that we don't pad
        placeholder or user visible output's strides.

        Add a matmul in the beginning and end so we can pad strides for
        intermediate tensors.
        """

        def f(x, y):
            x = torch.matmul(x, y)
            x = x + 1
            return torch.matmul(x, y)

        x = torch.randn(2, 16, 127)
        y = torch.randn(127, 127)
        self.common_numeric_check(f, x, y)

    @maybe_cprofile
    def run_acc_and_perf_test(self, model, inputs, perf_inputs=None, tol=1e-3):
        if perf_inputs is None:
            perf_inputs = inputs

        def _process_inputs(x):
            """
            return args and kwargs
            """
            if isinstance(x, dict):
                return [], x

            if not isinstance(inputs, (tuple, list)):
                x = [x]

            return x, {}

        args, kwargs = _process_inputs(inputs)
        perf_args, perf_kwargs = _process_inputs(perf_inputs)

        if DO_ACC_TEST:
            model.eval()
            self.common_numeric_check(model, *args, **kwargs, tol=tol)
        else:
            print("Accuracy test skipped")

        model.train()

        if DO_PERF_TEST:
            print("Do performance test")

            if len(kwargs) > 0:
                # for huggingface models
                def get_f(m, optim):
                    def f(*args, **kwargs):
                        optim.zero_grad(True)
                        with torch.cuda.amp.autocast():
                            pred = m(*args, **kwargs)
                            loss = pred[0]
                        loss.backward()
                        optim.step()
    
                    return f
            else:
                def get_f(m, optim):
                    def f(*args, **kwargs):
                        optim.zero_grad(True)
                        with torch.cuda.amp.autocast():
                            pred = m(*args, **kwargs)
                            if type(pred).__name__ in ("SequenceClassifierOutput"):
                                pred = pred.logits
                            elif isinstance(pred, dict):
                                pred = pred["loss"]
                            elif isinstance(pred, torch.Tensor):
                                pass
                            else:
                                raise NotImplementedError("unexpected model output")
                            loss = pred.sum()
                        loss.backward()
                        optim.step()
    
                    return f

            latency_with_padding = None
            print("Benchmark with padding")
            with config.patch(
                comprehensive_padding=True
            ):
                m_copy_with_padding = copy.deepcopy(model)
                optim_with_padding = get_optim(m_copy_with_padding)
                opt_f_with_padding = torch.compile(get_f(m_copy_with_padding, optim_with_padding))
                latency_with_padding = do_bench(lambda: opt_f_with_padding(*perf_args, **perf_kwargs))
            latency_without_padding = None
            print("bencmark without padding")
            with config.patch(comprehensive_padding=False):
                m_copy_without_padding = copy.deepcopy(model)
                optim_without_padding = get_optim(m_copy_without_padding)
                opt_f_without_padding = torch.compile(
                    get_f(m_copy_without_padding, optim_without_padding)
                )
                latency_without_padding = do_bench(
                    lambda: opt_f_without_padding(*perf_args, **perf_kwargs)
                )
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            # profiling
            self.do_profiling(opt_f_with_padding, opt_f_without_padding, args=perf_args, kwargs=perf_kwargs)

    def do_profiling(self, f_lhs, f_rhs, tag_lhs="With padding", tag_rhs="Without padding", args=(), kwargs={}):
       torch.cuda.synchronize()
       with torch.profiler.profile(with_stack=WITH_STACK) as p:
           niter = 3
           for _ in range(niter):
               with torch.profiler.record_function(
                   tag_lhs
               ):
                   f_lhs(*args, **kwargs)

               with torch.profiler.record_function(tag_rhs):
                   f_rhs(*args, **kwargs)
           torch.cuda.synchronize()

       profile_path = "/tmp/chrome.json"
       p.export_chrome_trace(profile_path)
       print(f"Chrome trace is written to {profile_path}")
       # breakpoint()

    def test_pytorch_unet(self):
        """
        Use adam as the optimizer.
        """
        import sys
        sys.path.append("/home/shunting/ws/pytorch/torchbenchmark/torchbenchmark/models/pytorch_unet/pytorch_unet/unet/")  # XXX hack, will cleanup
        from unet_model import UNet
        model = UNet(n_channels=3, n_classes=2, bilinear=True)
        if os.environ.get("SVAR") == "1":
            inputs = (
                torch.randn(1, 3, 640, 959), # x
                torch.randn(1, 64, 640, 959), # x1
                torch.randn(1, 128, 320, 479), # x2
                torch.randn(1, 256, 160, 239), # x3
                torch.randn(1, 512, 80, 119), # x4 
                torch.randn(1, 512, 40, 59), # x5
            )
        else:
            inputs = torch.randn(1, 3, 640, 959)

        self.run_acc_and_perf_test(model, inputs)

    def test_hf_Whisper(self):
        from transformers import WhisperConfig, AutoModelForAudioClassification
        config = WhisperConfig()
        model = AutoModelForAudioClassification.from_config(config)
        bs = 8
        feature_size = 80
        seq_length = 3000
        inputs = torch.randn(bs, feature_size, seq_length)
        self.run_acc_and_perf_test(model, (inputs,))

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_BertForMaskedLM(self, num_layers=1):
        """
        Make sure padding can be 'almost' as good as using a good shape.
        """
        from transformers import BertForMaskedLM 
        config_cls = BertForMaskedLM.config_class
        bs = 16
        seq_length = 512

        def create_model(vocab_size):
            config = config_cls()
            config.num_hidden_layers = num_layers
            config.vocab_size = vocab_size
            inputs = gen_transformer_inputs(config.vocab_size, bs, seq_length)
            model = BertForMaskedLM(config)

            optim = get_optim(model)
            
            def f(**inputs):
                optim.zero_grad(True)
                with torch.cuda.amp.autocast():
                    pred = model(**inputs)
                    loss = pred[0]
                loss.backward()
                optim.step()

            return torch.compile(f), inputs

        f_good_shape, inputs_good_shape = create_model(30528)
        f_bad_shape, inputs_bad_shape = create_model(30522)

        print("benchmark for good shape")
        latency_good_shape = do_bench(lambda: f_good_shape(**inputs_good_shape))
        print("benchmark for bad shape")
        latency_bad_shape = do_bench(lambda: f_bad_shape(**inputs_bad_shape))
        print(
            f"Latency with good and bad shape: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}"
        )

        self.do_profiling(lambda: f_good_shape(**inputs_good_shape), lambda: f_bad_shape(**inputs_bad_shape), tag_lhs="With good shape", tag_rhs="With bad shape")


    def test_longformer(self, bs=4):
        from transformers import AutoConfig, AutoModelForMaskedLM
        config = AutoConfig.from_pretrained("allenai/longformer-base-4096")
        model = AutoModelForMaskedLM.from_config(config)

        # input
        #   "input_ids": [4, 1024]
        #   "labels": [4, 1024]
        vocab_size = model.config.vocab_size
        seq_length = 1024
        input_dict = gen_transformer_inputs(vocab_size, bs, seq_length)

        self.run_acc_and_perf_test(model, input_dict)

    def test_longformer_small_bs(self):
        """
        The model exists in both HF and TB. In TB it uses a samller batch size.
        """
        self.test_longformer(bs=2)

    def test_rexnet(self):
        inputs = torch.randn([128, 3, 224, 224])
        model = create_timm_model("rexnet_100")
        self.run_acc_and_perf_test(model, inputs, tol=1e-2)
        
    def test_nvidia_deeprecommender(self):
        # SELU
        use_variant = True
        # layer_sizes = [197951, 512, 512, 1024, 512, 512, 197951]  # 9.955 v.s. 9.697
        # layer_sizes = [197952, 512, 512, 1024, 512, 512, 197951] # 8.446 v.s. 8.924
        # layer_sizes = [197952, 512, 512, 1024, 512, 512, 197952] # 7.201 v.s. 7.216
        # XXX also check this since the loss is even larger than ReLU
        # layer_sizes = [197951, 512, 512, 1024, 512, 512, 197952] # 8.713 v.s. 7.997

        # ReLU
        layer_sizes = [197951, 512, 512, 1024, 512, 512, 197951] # 9.956 v.s. 9.939
        # layer_sizes = [197952, 512, 512, 1024, 512, 512, 197951] # 8.369 v.s. 8.882
        # layer_sizes = [197952, 512, 512, 1024, 512, 512, 197952] # 7.156 v.s. 7.171
        # layer_sizes = [197951, 512, 512, 1024, 512, 512, 197952] # 8.766 v.s. 8.228

        x = torch.randn(4, layer_sizes[0])

        class Model(nn.Module):
            def __init__(self, use_variant=True):
                super().__init__()
                mod_list = []
                for i in range(len(layer_sizes) - 1):
                    mod_list.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                    if use_variant:
                        mod_list.append(nn.ReLU())
                    else:
                        mod_list.append(nn.SELU())

                    if i == 2:
                        mod_list.append(nn.Dropout(0.8))
                self.seq = nn.Sequential(*mod_list)

            def forward(self, x):
                return self.seq(x)

        m = Model(use_variant=use_variant)

        perf_inputs = torch.randn(256, layer_sizes[0])
        self.run_acc_and_perf_test(m, x, perf_inputs)

    def test_efficient_det(self):
        import sys
        # Will cleanup
        sys.path.append("/home/shunting/ws/pytorch/torchbenchmark")

        from torchbenchmark.models import timm_efficientdet
        try:
            torch.set_default_device("cpu")
            benchmark = timm_efficientdet.Model(
                test="train",
                device="cuda",
                batch_size=1,
            )
        finally:
            torch.set_default_device("cuda")
        model, inputs = benchmark.get_module()

        self.run_acc_and_perf_test(model, inputs)

    def skip_test_efficient_det(self):
        """
        Tried to construct the model and input using effdet package directly.
        But there are too many arguments to setup..
        """
        from effdet import create_model, create_loader, create_dataset

        model = create_model(
            model_name="tf_efficientdet_d1",
            bench_task="train",
            num_classes=None,
            pretrained=False,
            pretrained_backbone=True,
            redundant_bias=None,
            label_smoothing=None,
            legacy_focal=None,
            jit_loss=None,
            soft_nms=None,
            bench_labeler=False,
            checkpoint_path="",
        )
        dataset_train, _ = create_dataset(
            "coco",
            root=root,
            custom_dataset_cfg=Coco2017MinimalCfg())
        loader = create_loader(
        )
        pass

    def test_conv(self):
        x_shape = (1, 128, 640, 959)
        x1 = torch.randn(*x_shape)
        padded_stride = ir.Layout._pad_strides(x1.stride(), x1.shape)

        x2 = rand_strided(x_shape, padded_stride, device="cuda")
        x2.copy_(x1)

        weight = torch.randn(64, 128, 3, 3)

        def fun(x, weight):
            return torch.convolution(x, weight, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        ref = fun(x1, weight)
        act = fun(x2, weight)
        self.check_close(ref, act)
        if DO_PERF_TEST:
            latency_with_padding = do_bench(lambda: fun(x2, weight))
            latency_without_padding = do_bench(lambda: fun(x1, weight))
            print(
                f"Latency with and without padding: {latency_with_padding:.3f} v.s. {latency_without_padding:.3f}"
            )

            # According to this chrome trace: https://gist.github.com/shunting314/ce45398f7d51a63ce05fc8d411faddb3
            # An extra kernel is called if we pad x1 here. That cuase perf loss.
            self.do_profiling(lambda: fun(x2, weight), lambda: fun(x1, weight))

    @unittest.skipIf(not DO_PERF_TEST, "Perf test not enabled")
    def test_matmul(self):
        x_good_shape = torch.randn(8192, 30528, dtype=torch.float16)
        weight_good_shape = torch.randn(30528, 768, dtype=torch.float16)
        out_good_shape = torch.randn(8192, 768, dtype=torch.float16)

        x_bad_shape = rand_strided((8192, 30522), (30528, 1), device="cuda", dtype=torch.float16)
        # x_bad_shape = rand_strided((8192, 30522), (30522, 1), device="cuda", dtype=torch.float16)
        weight_bad_shape = torch.randn(30522, 768, dtype=torch.float16)
        out_bad_shape = torch.randn(8192, 768, dtype=torch.float16)

        def f(x, weight, out):
            torch.mm(x, weight, out=out)
            return out

        f1 = torch.compile(functools.partial(f, x_good_shape, weight_good_shape, out_good_shape))
        # f1 = lambda: 0
        f2 = torch.compile(functools.partial(f, x_bad_shape, weight_bad_shape, out_bad_shape))
        latency_good_shape = do_bench(f1)
        latency_bad_shape = do_bench(f2)
        # Latency with good and bad shapes: 1.705 v.s. 2.625
        print(f"Latency with good and bad shapes: {latency_good_shape:.3f} v.s. {latency_bad_shape:.3f}")
        self.do_profiling(f1, f2)

    def test_padmm(self):
        mat1_pad = torch.randn(8192, 30522, dtype=torch.float16)
        mat2_pad = torch.randn(30522, 768, dtype=torch.float16)

        def f():
            return mat1_pad @ mat2_pad
            
        def pad_dim(x: Tensor, padded_length: int, dim: int) -> Tensor:
            pad = x.new_zeros(*x.shape[:dim], padded_length, *x.shape[dim + 1 :])
            return torch.cat([x, pad], dim=dim)


        @torch.compile(fullgraph=True, options={"triton.cudagraphs": False})
        def g():
            mat1 = mat1_pad
            mat2 = mat2_pad
            # return pad_mm(mat1_pad, mat2_pad, 0, 6, 0)
            mat1 = pad_dim(mat1, 6, 1)
            mat2 = pad_dim(mat2, 6, 0)
            return torch.ops.aten.mm(mat1, mat2)

        ori_time = do_bench(f)
        pad_time = do_bench(g)

        # test_matmul result: 1.750 v.s. 2.626
        # this result: 2.616 v.s. 3.374
        # New result: 2.617 v.s. 2.947
        # 2.617 v.s. 2.257 Now!
        print(f"Latency between origional matmul and padded matmul: {ori_time:.3f} v.s. {pad_time:.3f}")
        self.do_profiling(f, g, "No MM Padding", "With mm padding")

    def test_cat(self):
        """
        Compare the perf between aten cat and compiled cat.

        Latency between eager and compiled: 1.596 v.s. 0.601

        Eager cat can be 2.66x slower than inductor kernel.
        """
        x = torch.randn(8192, 30522, dtype=torch.float16)
        def f(x):
            pad = x.new_zeros(x.size(0), 6)
            return torch.cat([x, pad], dim=1)

        # disable cudagraphs since cudagraphs need copy the input which
        # distort the latency a lot! (double the latency here for compiled
        # version)
        with config.patch("triton.cudagraphs", False):
            opt_f = torch.compile(f)
            opt_f(x)
        eager_time = do_bench(lambda: f(x))
        opt_time = do_bench(lambda: opt_f(x))
        print(f"Latency between eager and compiled: {eager_time:.3f} v.s. {opt_time:.3f}")
        self.do_profiling(lambda: f(x), lambda: opt_f(x), "Eager Cat", "Compiled Cat")
        
        

if __name__ == "__main__":
    if HAS_CUDA:
        torch.set_float32_matmul_precision("high")
        torch.set_default_device("cuda")
        run_tests()
