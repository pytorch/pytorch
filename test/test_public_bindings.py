# Owner(s): ["module: autograd"]

from torch.testing._internal.common_utils import TestCase, run_tests

import torch

import pkgutil
import importlib

from types import ModuleType

try:
    import tensorrt as trt
except Exception:
    trt = {}

class TestPublicBindings(TestCase):
    def test_valid_module_attribute(self):
        # No new entry should EVER be added to this set, only removed!
        missing_all_attr_allowlist = set([
            'torch.ao', 'torch.ao.nn', 'torch.ao.nn.sparse', 'torch.ao.ns',
            'torch.ao.ns.fx', 'torch.ao.quantization', 'torch.ao.quantization.fx',
            'torch.ao.quantization.fx.backend_config_dict', 'torch.ao.sparsity',
            'torch.ao.sparsity.experimental', 'torch.ao.sparsity.experimental.pruner',
            'torch.ao.sparsity.scheduler', 'torch.ao.sparsity.sparsifier', 'torch.backends',
            'torch.backends.cuda', 'torch.backends.cudnn', 'torch.backends.mkl', 'torch.backends.mkldnn',
            'torch.backends.openmp', 'torch.backends.quantized', 'torch.backends.xnnpack', 'torch.contrib',
            'torch.cpu', 'torch.cpu.amp', 'torch.cuda', 'torch.cuda.amp', 'torch.distributed',
            'torch.distributed.algorithms', 'torch.distributed.algorithms.ddp_comm_hooks',
            'torch.distributed.algorithms.model_averaging', 'torch.distributed.autograd',
            'torch.distributed.elastic', 'torch.distributed.elastic.agent', 'torch.distributed.elastic.agent.server',
            'torch.distributed.elastic.events', 'torch.distributed.elastic.metrics',
            'torch.distributed.elastic.multiprocessing', 'torch.distributed.elastic.multiprocessing.errors',
            'torch.distributed.elastic.rendezvous', 'torch.distributed.elastic.timer',
            'torch.distributed.elastic.utils', 'torch.distributed.elastic.utils.data', 'torch.distributed.launcher',
            'torch.distributed.nn', 'torch.distributed.optim', 'torch.distributed.pipeline',
            'torch.distributed.pipeline.sync', 'torch.distributed.rpc', 'torch.for_onnx', 'torch.futures', 'torch.fx',
            'torch.fx.experimental', 'torch.fx.experimental.fx2trt', 'torch.fx.experimental.fx_acc',
            'torch.fx.experimental.unification', 'torch.fx.experimental.unification.multipledispatch',
            'torch.fx.passes', 'torch.jit', 'torch.jit.mobile', 'torch.linalg', 'torch.nn', 'torch.nn.backends',
            'torch.nn.intrinsic', 'torch.nn.intrinsic.qat', 'torch.nn.intrinsic.quantized',
            'torch.nn.intrinsic.quantized.dynamic', 'torch.nn.qat', 'torch.nn.quantizable', 'torch.nn.quantized',
            'torch.nn.quantized.dynamic', 'torch.nn.utils', 'torch.onnx', 'torch.optim', 'torch.package',
            'torch.package.analyze', 'torch.profiler', 'torch.quantization', 'torch.quantization.fx', 'torch.testing',
            'torch.utils', 'torch.utils.backcompat', 'torch.utils.benchmark', 'torch.utils.benchmark.examples',
            'torch.utils.benchmark.op_fuzzers', 'torch.utils.benchmark.utils',
            'torch.utils.benchmark.utils.valgrind_wrapper', 'torch.utils.bottleneck', 'torch.utils.data.communication',
            'torch.utils.data.datapipes', 'torch.utils.data.datapipes.utils', 'torch.utils.ffi', 'torch.utils.hipify',
            'torch.utils.model_dump', 'torch.utils.tensorboard',
        ])

        # No new entry should EVER be added to this set, only removed!
        missing_module_attr_allowlist = set([
            'torch.distributions.Poisson', 'torch.utils.data.datapipes.iter.FileLister',
            'torch.nn.modules.BatchNorm2d', 'torch.nn.modules.CrossEntropyLoss',
            'torch.nn.intrinsic.modules.ConvReLU2d', 'torch.nn.modules.BatchNorm3d', 'torch.nn.modules.Fold',
            'torch.utils.data.datapipes.iter.BucketBatcher', 'torch.utils.data.datapipes.iter.Demultiplexer',
            'torch.nn.modules.LazyConv2d', 'torch.nn.modules.CELU', 'torch.nn.quantized.dynamic.modules.GRUCell',
            'torch.nn.intrinsic.qat.modules.LinearReLU', 'torch.nn.modules.RNNCell',
            'torch.distributions.IndependentTransform', 'torch.fft.fft', 'torch.nn.modules.ParameterDict',
            'torch.special.softmax', 'torch.utils.data.MapDataPipe', 'torch.nn.modules.LSTMCell',
            'torch.nn.quantized.modules.Conv3d', 'torch.special.polygamma', 'torch.nn.modules.Dropout',
            'torch.nn.modules.LogSigmoid', 'torch.multiprocessing.set_executable',
            'torch.nn.quantized.modules.MaxPool2d', 'torch.nn.quantized.modules.InstanceNorm1d',
            'torch.distributions.Geometric', 'torch.distributions.Multinomial', 'torch.multiprocessing.Semaphore',
            'torch.distributions.StudentT', 'torch.utils.data.datapipes.iter.FileLoader', 'torch.special.xlogy',
            'torch.fft.rfftfreq', 'torch.nn.modules.TransformerEncoder', 'torch.distributions.Gamma',
            'torch.nn.modules.AlphaDropout', 'torch.nn.modules.LazyBatchNorm2d', 'torch.nn.modules.AdaptiveAvgPool3d',
            'torch.distributions.RelaxedOneHotCategorical', 'torch.multiprocessing.get_context',
            'torch.utils.data.DataLoader2', 'torch.multiprocessing.Pipe', 'torch.multiprocessing.Queue',
            'torch.distributions.Chi2', 'torch.nn.modules.Unfold', 'torch.nn.quantizable.modules.MultiheadAttention',
            'torch.distributions.OneHotCategoricalStraightThrough', 'torch.fft.rfftn', 'torch.special.ndtr',
            'torch.nn.modules.ReLU6', 'torch.nn.modules.AdaptiveLogSoftmaxWithLoss',
            'torch.nn.quantized.modules.Sigmoid', 'torch.nn.modules.Embedding',
            'torch.ao.nn.sparse.quantized.dynamic.Linear', 'torch.distributions.Categorical',
            'torch.nn.modules.Module', 'torch.nn.quantized.modules.LeakyReLU', 'torch.special.log_softmax',
            'torch.distributions.Bernoulli', 'torch.distributions.identity_transform', 'torch.multiprocessing.Lock',
            'torch.nn.quantized.modules.BatchNorm3d', 'torch.nn.modules.RNNCellBase', 'torch.special.psi',
            'torch.nn.parallel.parallel_apply', 'torch.nn.quantized.modules.ReLU6', 'torch.distributions.Transform',
            'torch.multiprocessing.Barrier', 'torch.multiprocessing.current_process', 'torch.multiprocessing.Event',
            'torch.nn.modules.Flatten', 'torch.nn.intrinsic.modules.ConvBnReLU1d', 'torch.nn.modules.SmoothL1Loss',
            'torch.nn.intrinsic.qat.modules.ConvBnReLU1d',
            'torch.utils.data.datapipes.dataframe.DataFramesAsTuplesPipe', 'torch.nn.modules.FeatureAlphaDropout',
            'torch.nn.modules.FractionalMaxPool2d', 'torch.multiprocessing.RawValue', 'torch.nn.modules.ELU',
            'torch.autograd.Function', 'torch.nn.modules.LSTM', 'torch.utils.data.datapipes.iter.StreamReader',
            'torch.nn.modules.ReplicationPad2d', 'torch.nn.modules.PReLU', 'torch.nn.modules.MarginRankingLoss',
            'torch.distributions.AffineTransform', 'torch.nn.modules.Bilinear', 'torch.distributions.Pareto',
            'torch.nn.modules.TransformerDecoder', 'torch.nn.modules.GRUCell', 'torch.nn.parallel.DataParallel',
            'torch.nn.modules.Container', 'torch.multiprocessing.Condition',
            'torch.multiprocessing.get_all_start_methods', 'torch.special.gammaln', 'torch.nn.modules.LogSoftmax',
            'torch.distributions.TransformedDistribution', 'torch.distributions.transform_to', 'torch.nn.modules.GELU',
            'torch.nn.quantized.modules._ConvNd', 'torch.nn.modules.LPPool2d', 'torch.distributions.Uniform',
            'torch.special.log1p', 'torch.distributions.ContinuousBernoulli', 'torch.utils.data.non_deterministic',
            'torch.nn.modules.NLLLoss2d', 'torch.multiprocessing.get_start_method', 'torch.nn.modules.Sigmoid',
            'torch.multiprocessing.SimpleQueue', 'torch.multiprocessing.active_children', 'torch.nn.modules.LayerNorm',
            'torch.distributions.NegativeBinomial', 'torch.multiprocessing.Process', 'torch.nn.modules.Hardsigmoid',
            'torch.multiprocessing.allow_connection_pickling', 'torch.distributions.biject_to',
            'torch.distributions.CorrCholeskyTransform', 'torch.utils.data.datapipes.iter.Forker',
            'torch.distributions.TanhTransform', 'torch.nn.modules.ReLU', 'torch.distributions.OneHotCategorical',
            'torch.utils.data.functional_datapipe', 'torch.nn.intrinsic.quantized.modules.BNReLU2d',
            'torch.nn.modules.LazyBatchNorm1d', 'torch.nn.modules.GLU', 'torch.nn.quantized.modules.Conv2d',
            'torch.nn.modules.ParameterList', 'torch.nn.modules.AdaptiveMaxPool2d', 'torch.nn.modules.Softmin',
            'torch.ao.nn.sparse.quantized.LinearPackedParams', 'torch.utils.data.ChainDataset',
            'torch.nn.modules.AvgPool3d', 'torch.special.logsumexp', 'torch.distributions.StackTransform',
            'torch.nn.modules.RNNBase', 'torch.nn.quantized.modules.ELU', 'torch.utils.data.Dataset',
            'torch.nn.modules.Hardtanh', 'torch.nn.modules.TransformerDecoderLayer', 'torch.nn.modules.PixelShuffle',
            'torch.nn.modules.SoftMarginLoss', 'torch.nn.intrinsic.quantized.modules.ConvReLU1d',
            'torch.nn.modules.Threshold', 'torch.multiprocessing.Array', 'torch.nn.modules.InstanceNorm3d',
            'torch.distributions.HalfNormal', 'torch.nn.modules.AdaptiveAvgPool2d', 'torch.nn.qat.modules.Conv2d',
            'torch.utils.data.TensorDataset', 'torch.multiprocessing.Pool', 'torch.multiprocessing.BufferTooShort',
            'torch.nn.modules.CTCLoss', 'torch.distributions.AbsTransform', 'torch.distributions.SigmoidTransform',
            'torch.fft.irfftn', 'torch.utils.data.datapipes.iter.UnBatcher', 'torch.nn.modules.GaussianNLLLoss',
            'torch.special.zeta', 'torch.nn.modules.LazyLinear', 'torch.nn.modules.BatchNorm1d',
            'torch.utils.data.SequentialSampler', 'torch.special.erfinv', 'torch.multiprocessing.freeze_support',
            'torch.distributions.ReshapeTransform', 'torch.multiprocessing.Value', 'torch.utils.data.RandomSampler',
            'torch.nn.modules.ConvTranspose3d', 'torch.nn.modules.MaxPool1d', 'torch.nn.modules.ChannelShuffle',
            'torch.utils.data.datapipes.iter.Shuffler', 'torch.special.entr',
            'torch.utils.data.datapipes.iter.ZipArchiveReader', 'torch.nn.modules.PairwiseDistance',
            'torch.special.i1e', 'torch.distributions.CatTransform', 'torch.nn.modules.ConstantPad1d',
            'torch.nn.modules.MaxPool3d', 'torch.nn.modules.MaxPool2d', 'torch.nn.quantized.modules.InstanceNorm3d',
            'torch.distributions.kl_divergence', 'torch.nn.modules.PoissonNLLLoss',
            'torch.utils.data.datapipes.iter.Batcher', 'torch.multiprocessing.RLock',
            'torch.distributions.Kumaraswamy', 'torch.nn.modules.GroupNorm', 'torch.nn.modules.RReLU',
            'torch.special.xlog1py', 'torch.special.i1', 'torch.nn.quantized.dynamic.modules.GRU',
            'torch.nn.modules.Softshrink', 'torch.distributions.register_kl', 'torch.utils.data.argument_validation',
            'torch.nn.quantized.dynamic.modules.LSTM', 'torch.nn.modules.AdaptiveMaxPool1d', 'torch.nn.modules.Conv2d',
            'torch.nn.intrinsic.modules.ConvBn2d', 'torch.nn.modules.FractionalMaxPool3d',
            'torch.distributions.LowerCholeskyTransform', 'torch.nn.quantized.modules.ConvTranspose1d',
            'torch.distributions.VonMises', 'torch.nn.modules.Dropout3d', 'torch.utils.data.datapipes.iter.Zipper',
            'torch.utils.data.datapipes.iter.Mapper', 'torch.nn.modules.AdaptiveAvgPool1d',
            'torch.nn.modules.InstanceNorm2d', 'torch.nn.modules.Hardswish',
            'torch.utils.data.datapipes.iter.Collator', 'torch.nn.modules.SyncBatchNorm', 'torch.nn.modules.Unflatten',
            'torch.nn.modules.LPPool1d', 'torch.distributions.Binomial', 'torch.nn.intrinsic.qat.modules.ConvBn2d',
            'torch.ao.nn.sparse.quantized.Linear', 'torch.nn.modules.Softmax2d',
            'torch.nn.intrinsic.modules.ConvReLU3d', 'torch.utils.data.Sampler', 'torch.utils.data.IterDataPipe',
            'torch.nn.modules.MSELoss', 'torch.nn.modules.PixelUnshuffle', 'torch.nn.modules.LazyBatchNorm3d',
            'torch.nn.modules.AvgPool2d', 'torch.nn.modules.MultiLabelMarginLoss',
            'torch.nn.modules.LazyConvTranspose3d', 'torch.nn.modules.GRU', 'torch.nn.quantized.modules.Linear',
            'torch.fft.ifft', 'torch.nn.modules.MultiheadAttention', 'torch.distributions.Laplace',
            'torch.nn.quantizable.modules.LSTMCell', 'torch.nn.intrinsic.modules.BNReLU3d', 'torch.special.gammaincc',
            'torch.nn.modules.MaxUnpool2d', 'torch.utils.data.datapipes.iter.LineReader',
            'torch.nn.modules.ConvTranspose2d', 'torch.nn.modules.HingeEmbeddingLoss',
            'torch.nn.modules.LazyConvTranspose1d', 'torch.nn.intrinsic.modules.ConvBn1d',
            'torch.nn.modules.LocalResponseNorm', 'torch.fft.Tensor', 'torch.multiprocessing.TimeoutError',
            'torch.nn.modules.MultiMarginLoss', 'torch.distributions.FisherSnedecor', 'torch.special.i0',
            'torch.utils.data.datapipes.iter.IterableWrapper', 'torch.distributions.ExpTransform',
            'torch.special.multigammaln', 'torch.distributions.LowRankMultivariateNormal',
            'torch.nn.modules.LazyInstanceNorm2d', 'torch.nn.quantized.modules.FXFloatFunctional',
            'torch.multiprocessing.cpu_count', 'torch.fft.irfft', 'torch.nn.quantized.modules.QFunctional',
            'torch.utils.data.IterableDataset', 'torch.multiprocessing.set_forkserver_preload',
            'torch.nn.quantizable.modules.LSTM', 'torch.nn.modules.Sequential', 'torch.nn.modules.NLLLoss',
            'torch.nn.modules.Identity', 'torch.nn.modules.ModuleDict', 'torch.fft.hfft',
            'torch.distributions.LogisticNormal', 'torch.nn.modules.Softplus',
            'torch.nn.quantized.modules.ConvTranspose2d', 'torch.utils.data.datapipes.iter.Filter',
            'torch.utils.data.ConcatDataset', 'torch.special.round', 'torch.nn.modules.Transformer',
            'torch.nn.intrinsic.quantized.modules.LinearReLU', 'torch.nn.modules.AvgPool1d',
            'torch.distributions.Cauchy', 'torch.nn.quantized.dynamic.modules.Linear',
            'torch.multiprocessing.RawArray', 'torch.nn.intrinsic.qat.modules.ConvBnReLU3d',
            'torch.nn.quantized.modules.Embedding', 'torch.nn.quantized.modules.GroupNorm',
            'torch.nn.intrinsic.modules.BNReLU2d', 'torch.nn.parallel.DistributedDataParallel',
            'torch.nn.modules.Tanh', 'torch.nn.modules.Upsample', 'torch.multiprocessing.BoundedSemaphore',
            'torch.nn.modules.ModuleList', 'torch.nn.modules.LeakyReLU', 'torch.fft.fftshift',
            'torch.nn.modules.LazyConv3d', 'torch.special.erfcx', 'torch.multiprocessing.Manager',
            'torch.special.digamma', 'torch.nn.modules.InstanceNorm1d',
            'torch.utils.data.datapipes.iter.TarArchiveReader', 'torch.nn.modules.SiLU',
            'torch.utils.data.datapipes.map.SequenceWrapper', 'torch.nn.quantized.modules.Hardswish',
            'torch.distributions.StickBreakingTransform', 'torch.distributions.SoftmaxTransform',
            'torch.nn.intrinsic.modules.ConvBnReLU3d', 'torch.distributions.Beta',
            'torch.multiprocessing.set_start_method', 'torch.special.erf', 'torch.distributions.Gumbel',
            'torch.nn.intrinsic.modules.ConvBn3d', 'torch.nn.parallel.scatter', 'torch.fft.ifftshift',
            'torch.special.expm1', 'torch.nn.modules.Mish', 'torch.nn.modules.Conv1d',
            'torch.nn.quantized.dynamic.modules.LSTMCell', 'torch.distributions.LogNormal',
            'torch.nn.modules.L1Loss', 'torch.fft.fft2', 'torch.multiprocessing.get_logger',
            'torch.nn.modules.UpsamplingNearest2d', 'torch.nn.modules.ReplicationPad3d',
            'torch.nn.qat.modules.EmbeddingBag', 'torch.nn.modules.Linear', 'torch.nn.modules.ReflectionPad3d',
            'torch.nn.intrinsic.qat.modules.ConvBn1d', 'torch.nn.modules.ConvTranspose1d',
            'torch.distributions.MixtureSameFamily', 'torch.nn.intrinsic.quantized.modules.BNReLU3d',
            'torch.nn.intrinsic.qat.modules.ConvReLU3d', 'torch.multiprocessing.AuthenticationError',
            'torch.nn.quantized.modules.LayerNorm', 'torch.utils.data.datapipes.iter.RoutedDecoder',
            'torch.utils.data.runtime_validation_disabled', 'torch.distributions.Exponential',
            'torch.utils.data.datapipes.map.Mapper', 'torch.special.sinc', 'torch.nn.intrinsic.modules.ConvReLU1d',
            'torch.nn.modules.EmbeddingBag', 'torch.utils.data.Subset', 'torch.utils.data.DistributedSampler',
            'torch.fft.rfft2', 'torch.utils.data.datapipes.iter.Multiplexer',
            'torch.nn.intrinsic.qat.modules.freeze_bn_stats', 'torch.nn.modules.ZeroPad2d',
            'torch.utils.data.datapipes.iter.HttpReader', 'torch.utils.data.datapipes.map.Concater',
            'torch.nn.quantized.modules.BatchNorm2d', 'torch.distributions.MultivariateNormal',
            'torch.distributions.PowerTransform', 'torch.nn.intrinsic.quantized.modules.ConvReLU2d',
            'torch.nn.modules.TripletMarginWithDistanceLoss', 'torch.nn.modules.CosineEmbeddingLoss',
            'torch.multiprocessing.ProcessError', 'torch.nn.modules.ReplicationPad1d',
            'torch.distributions.RelaxedBernoulli', 'torch.nn.modules.RNN', 'torch.distributions.ExponentialFamily',
            'torch.distributions.Normal', 'torch.nn.modules.HuberLoss', 'torch.nn.modules.Softsign',
            'torch.nn.modules.TransformerEncoderLayer', 'torch.nn.modules.KLDivLoss',
            'torch.nn.modules.BCEWithLogitsLoss', 'torch.utils.data.random_split', 'torch.nn.modules.MaxUnpool1d',
            'torch.special.ndtri', 'torch.utils.data._DatasetKind', 'torch.utils.data.runtime_validation',
            'torch.nn.modules.LazyInstanceNorm3d', 'torch.multiprocessing.JoinableQueue',
            'torch.nn.intrinsic.modules.ConvBnReLU2d', 'torch.nn.modules.MaxUnpool3d', 'torch.nn.modules.Dropout2d',
            'torch.nn.modules.LazyConv1d', 'torch.nn.modules.Conv3d', 'torch.fft.ihfft', 'torch.fft.fftn',
            'torch.fft.irfft2', 'torch.utils.data.BatchSampler', 'torch.utils.data.get_worker_info',
            'torch.utils.data.WeightedRandomSampler', 'torch.nn.modules.MultiLabelSoftMarginLoss',
            'torch.nn.modules.UpsamplingBilinear2d', 'torch.nn.quantized.modules.ConvTranspose3d',
            'torch.nn.intrinsic.quantized.modules.ConvReLU3d', 'torch.special.gammainc',
            'torch.nn.intrinsic.qat.modules.ConvBn3d', 'torch.distributions.Distribution',
            'torch.utils.data.SubsetRandomSampler', 'torch.nn.intrinsic.modules._FusedModule',
            'torch.nn.modules.ReflectionPad2d', 'torch.nn.quantized.modules.FloatFunctional',
            'torch.nn.modules.AdaptiveMaxPool3d', 'torch.nn.parallel.data_parallel',
            'torch.nn.intrinsic.qat.modules.ConvBnReLU2d', 'torch.nn.parallel.replicate',
            'torch.nn.qat.modules.Conv3d', 'torch.utils.data.datapipes.iter.Sampler',
            'torch.nn.intrinsic.qat.modules.update_bn_stats', 'torch.nn.modules.ConstantPad2d',
            'torch.nn.intrinsic.qat.modules.ConvReLU2d', 'torch.nn.modules.SELU',
            'torch.utils.data.datapipes.iter.Concater', 'torch.distributions.Independent', 'torch.fft.rfft',
            'torch.utils.data.DataLoader', 'torch.utils.data.guaranteed_datapipes_determinism',
            'torch.nn.quantized.modules.Conv1d', 'torch.nn.quantized.dynamic.modules.RNNCell',
            'torch.multiprocessing.log_to_stderr', 'torch.autograd.Variable', 'torch.nn.modules.ConstantPad3d',
            'torch.fft.ifftn', 'torch.nn.quantized.modules.EmbeddingBag', 'torch.nn.modules.Tanhshrink',
            'torch.nn.qat.modules.Linear', 'torch.fft.ifft2', 'torch.fft.fftfreq',
            'torch.nn.intrinsic.modules.LinearReLU', 'torch.nn.quantized.modules.InstanceNorm2d',
            'torch.nn.modules.BCELoss', 'torch.distributions.Weibull', 'torch.nn.modules.Hardshrink',
            'torch.nn.intrinsic.quantized.dynamic.modules.LinearReLU', 'torch.distributions.LKJCholesky',
            'torch.distributions.Dirichlet', 'torch.utils.data.datapipes.dataframe.DFIterDataPipe',
            'torch.nn.modules.Softmax', 'torch.distributions.ComposeTransform', 'torch.nn.modules.CrossMapLRN2d',
            'torch.nn.modules.LazyConvTranspose2d', 'torch.nn.modules.TripletMarginLoss', 'torch.special.logit',
            'torch.special.expit', 'torch.nn.modules.CosineSimilarity', 'torch.special.erfc',
            'torch.utils.data.datapipes.iter.Grouper', 'torch.nn.parallel.gather', 'torch.nn.modules.ReflectionPad1d',
            'torch.special.exp2', 'torch.special.i0e', 'torch.nn.modules.LazyInstanceNorm1d',
            'torch.distributions.HalfCauchy',
        ])

        # No new entry should EVER be added to this set, only removed!
        missing_obj_attr_allowlist = set(['torch.utils.data.datapipes.iter.DFIterDataPipe'])

        def is_not_internal(modname):
            split_name = modname.split(".")
            for name in split_name:
                if name[0] == "_":
                    return False
            return True

        # Allow this script to run when
        #  - Built with USE_DISTRIBUTED=0
        #  - TensorRT is not installed
        #  - Until torch.utils.ffi module is removed
        def cannot_be_skipped(modname):
            if "distributed" in modname and not torch.distributed.is_available():
                return False
            if "fx2trt" in modname and not hasattr(trt, "__version__"):
                return False
            if modname == "torch.utils.ffi":
                return False
            return True

        missing_all_attr = set()
        missing_module_attr = set()
        missing_obj_attr = set()

        def error_handler(bad_pkg):
            if is_not_internal(bad_pkg) and cannot_be_skipped(bad_pkg):
                raise RuntimeError(f"Failed to import public package {bad_pkg}")

        for _, modname, ispkg in pkgutil.walk_packages(path=torch.__path__,
                                                       prefix=torch.__name__ + '.',
                                                       onerror=error_handler):
            if ispkg and is_not_internal(modname):

                try:
                    mod = importlib.import_module(modname)
                except Exception as e:
                    if cannot_be_skipped(modname):
                        raise e

                if not hasattr(mod, "__all__"):
                    missing_all_attr.add(modname)
                    continue
                for el_name in mod.__all__:
                    if not hasattr(mod, el_name):
                        missing_obj_attr.add(f"{modname}.{el_name}")
                        continue
                    obj = getattr(mod, el_name)
                    if isinstance(obj, ModuleType):
                        continue
                    if not hasattr(obj, "__module__") or obj.__module__ != modname:
                        missing_module_attr.add(f"{modname}.{el_name}")

        output = []

        # Generate error for missing `__all__` attribute on a module
        unexpected_missing = missing_all_attr - missing_all_attr_allowlist
        if unexpected_missing:
            mods = ", ".join(unexpected_missing)
            output.append(f"\nYou added the following module(s) to the PyTorch namespace '{mods}' "
                          "but they have no `__all__` attribute in a doc .rst file. You should use "
                          "this attribute to specify which functions are public.")
        unexpected_not_missing = missing_all_attr_allowlist - missing_all_attr
        if unexpected_not_missing:
            mods = ", ".join(unexpected_not_missing)
            output.append(f"\nThank you for adding the missing `__all__` for '{mods}', please update "
                          "the 'missing_all_attr_allowlist' in 'torch/test/test_public_bindings.py' by removing "
                          "the module(s) you fixed to make sure we do not regress on this in the future.")

        # Generate error for missing/wrong `__module__` attribute on a public API
        unexpected_missing = missing_module_attr - missing_module_attr_allowlist
        if unexpected_missing:
            mods = ", ".join(unexpected_missing)
            output.append(f"\nYou added the following function/class(es) to PyTorch '{mods}' "
                          "but they have no or incorrect `__module__` attribute. This attribute "
                          "must point to the module that exposes these functions as public API (as "
                          "defined using the `__all__` attribute on the module).")
        unexpected_not_missing = missing_module_attr_allowlist - missing_module_attr
        if unexpected_not_missing:
            mods = ", ".join(unexpected_not_missing)
            output.append(f"\nThank you for fixing the `__module__` attribute for '{mods}', please update "
                          "the 'missing_module_attr_allowlist' in 'torch/test/test_public_bindings.py' by "
                          "removing the function/class(es) you fixed to make sure we do not regress on this "
                          "in the future.")

        # Generate error for missing function/class that is listed in `__all__`
        unexpected_missing = missing_obj_attr - missing_obj_attr_allowlist
        if unexpected_missing:
            mods = ", ".join(unexpected_missing)
            output.append(f"\nYou added the following function/class(es) to PyTorch '{mods}' "
                          "but, while they are part of the list of public APIs for their module "
                          "(as described by `__all__`), this object does not exist on this module.")
        unexpected_not_missing = missing_obj_attr_allowlist - missing_obj_attr
        if unexpected_not_missing:
            mods = ", ".join(unexpected_not_missing)
            output.append(f"\nThank you for fixing the existence of '{mods}', please update the "
                          "'missing_obj_attr_allowlist' in 'torch/test/test_public_bindings.py' by "
                          "removing the function/class(es) you fixed to make sure we do not regress on this "
                          "in the future.")

        self.assertFalse(output, msg="Some Error where detected in the namespace attributes: " + "\n".join(output))

    def test_no_new_bindings(self):
        """
        This test aims to stop the introduction of new JIT bindings into torch._C
        whose names do not start with _. Such bindings are made available as
        torch.XXX, which may not be desirable.

        If your change causes this test to fail, add your new binding to a relevant
        submodule of torch._C, such as torch._C._jit (or other relevant submodule of
        torch._C). If your binding really needs to be available as torch.XXX, add it
        to torch._C and add it to the allowlist below.

        If you have removed a binding, remove it from the allowlist as well.
        """
        # This allowlist contains every binding in torch._C that is copied into torch at
        # the time of writing. It was generated with
        #
        #   {elem for elem in dir(torch._C) if not elem.startswith("_")}
        #
        torch_C_allowlist_superset = {
            "AggregationType",
            "AliasDb",
            "AnyType",
            "Argument",
            "ArgumentSpec",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "AVG",
            "BenchmarkConfig",
            "BenchmarkExecutionStats",
            "BFloat16StorageBase",
            "Block",
            "BoolStorageBase",
            "BoolType",
            "BufferDict",
            "ByteStorageBase",
            "CallStack",
            "Capsule",
            "CharStorageBase",
            "ClassType",
            "clear_autocast_cache",
            "Code",
            "CompilationUnit",
            "CompleteArgumentSpec",
            "ComplexDoubleStorageBase",
            "ComplexFloatStorageBase",
            "ComplexType",
            "ConcreteModuleType",
            "ConcreteModuleTypeBuilder",
            "CONV_BN_FUSION",
            "cpp",
            "CudaBFloat16StorageBase",
            "CudaBFloat16TensorBase",
            "CudaBFloat16TensorBase",
            "CudaBoolStorageBase",
            "CudaBoolTensorBase",
            "CudaBoolTensorBase",
            "CudaByteStorageBase",
            "CudaByteTensorBase",
            "CudaByteTensorBase",
            "CudaCharStorageBase",
            "CudaCharTensorBase",
            "CudaCharTensorBase",
            "CudaComplexDoubleStorageBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexDoubleTensorBase",
            "CudaComplexFloatStorageBase",
            "CudaComplexFloatTensorBase",
            "CudaComplexFloatTensorBase",
            "CudaDoubleStorageBase",
            "CudaDoubleTensorBase",
            "CudaDoubleTensorBase",
            "CudaFloatStorageBase",
            "CudaFloatTensorBase",
            "CudaHalfStorageBase",
            "CudaHalfTensorBase",
            "CudaIntStorageBase",
            "CudaIntTensorBase",
            "CudaIntTensorBase",
            "CudaLongStorageBase",
            "CudaLongTensorBase",
            "CudaLongTensorBase",
            "CudaShortStorageBase",
            "CudaShortTensorBase",
            "CudaShortTensorBase",
            "DeepCopyMemoTable",
            "default_generator",
            "DeserializationStorageContext",
            "device",
            "DeviceObjType",
            "DictType",
            "DisableTorchFunction",
            "DoubleStorageBase",
            "dtype",
            "EnumType",
            "ErrorReport",
            "ExecutionPlan",
            "FatalError",
            "FileCheck",
            "finfo",
            "FloatStorageBase",
            "FloatType",
            "fork",
            "FunctionSchema",
            "FUSE_ADD_RELU",
            "Future",
            "FutureType",
            "Generator",
            "get_autocast_cpu_dtype",
            "get_default_dtype",
            "get_num_interop_threads",
            "get_num_threads",
            "Gradient",
            "Graph",
            "GraphExecutorState",
            "HalfStorageBase",
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mlc",
            "has_openmp",
            "has_spectral",
            "HOIST_CONV_PACKED_PARAMS",
            "iinfo",
            "import_ir_module_from_buffer",
            "import_ir_module",
            "InferredType",
            "init_num_threads",
            "INSERT_FOLD_PREPACK_OPS",
            "InterfaceType",
            "IntStorageBase",
            "IntType",
            "IODescriptor",
            "is_anomaly_enabled",
            "is_autocast_cache_enabled",
            "is_autocast_cpu_enabled",
            "is_autocast_enabled",
            "is_grad_enabled",
            "is_inference_mode_enabled",
            "JITException",
            "layout",
            "ListType",
            "LiteScriptModule",
            "LockingLogger",
            "LoggerBase",
            "LongStorageBase",
            "memory_format",
            "merge_type_from_type_comment",
            "MobileOptimizerType",
            "ModuleDict",
            "Node",
            "NoneType",
            "NoopLogger",
            "NumberType",
            "OperatorInfo",
            "OptionalType",
            "ParameterDict",
            "parse_ir",
            "parse_schema",
            "parse_type_comment",
            "PyObjectType",
            "PyTorchFileReader",
            "PyTorchFileWriter",
            "QInt32StorageBase",
            "QInt8StorageBase",
            "qscheme",
            "QUInt4x2StorageBase",
            "QUInt2x4StorageBase",
            "QUInt8StorageBase",
            "read_vitals",
            "REMOVE_DROPOUT",
            "RRefType",
            "ScriptClass",
            "ScriptClassFunction",
            "ScriptDict",
            "ScriptDictIterator",
            "ScriptDictKeyIterator",
            "ScriptList",
            "ScriptListIterator",
            "ScriptFunction",
            "ScriptMethod",
            "ScriptModule",
            "ScriptModuleSerializer",
            "ScriptObject",
            "ScriptObjectProperty",
            "SerializationStorageContext",
            "set_anomaly_enabled",
            "set_autocast_cache_enabled",
            "set_autocast_cpu_dtype",
            "set_autocast_cpu_enabled",
            "set_autocast_enabled",
            "set_flush_denormal",
            "set_num_interop_threads",
            "set_num_threads",
            "set_vital",
            "ShortStorageBase",
            "Size",
            "StaticModule",
            "Stream",
            "StreamObjType",
            "StringType",
            "SUM",
            "TensorType",
            "ThroughputBenchmark",
            "TracingState",
            "TupleType",
            "Type",
            "unify_type_list",
            "UnionType",
            "Use",
            "Value",
            "autocast_decrement_nesting",
            "autocast_increment_nesting",
            "clear_autocast_cache",
            "cpp",
            "default_generator",
            "device",
            "dtype",
            "finfo",
            "fork",
            "get_default_dtype",
            "get_num_interop_threads",
            "get_num_threads",
            "has_cuda",
            "has_cudnn",
            "has_lapack",
            "has_mkl",
            "has_mkldnn",
            "has_mlc",
            "has_openmp",
            "iinfo",
            "import_ir_module",
            "import_ir_module_from_buffer",
            "init_num_threads",
            "is_anomaly_enabled",
            "is_autocast_enabled",
            "is_grad_enabled",
            "layout",
            "memory_format",
            "merge_type_from_type_comment",
            "parse_ir",
            "parse_schema",
            "parse_type_comment",
            "qscheme",
            "set_anomaly_enabled",
            "set_autocast_enabled",
            'set_autocast_gpu_dtype',
            'get_autocast_gpu_dtype',
            "set_flush_denormal",
            "set_num_interop_threads",
            "set_num_threads",
            "unify_type_list",
            "vitals_enabled",

            "wait",
        }
        torch_C_bindings = {elem for elem in dir(torch._C) if not elem.startswith("_")}

        # Check that the torch._C bindings are all in the allowlist. Since
        # bindings can change based on how PyTorch was compiled (e.g. with/without
        # CUDA), the two may not be an exact match but the bindings should be
        # a subset of the allowlist.
        difference = torch_C_bindings.difference(torch_C_allowlist_superset)
        msg = f"torch._C had bindings that are not present in the allowlist:\n{difference}"
        self.assertTrue(torch_C_bindings.issubset(torch_C_allowlist_superset), msg)


if __name__ == '__main__':
    run_tests()
