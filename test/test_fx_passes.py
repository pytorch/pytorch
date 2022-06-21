# Owner(s): ["oncall: fx"]

import copy
import pickle
import operator
import importlib

import torch
from torch.fx._symbolic_trace import symbolic_trace

from torch.fx.partitioner.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.partitioner.nvfuser_operator_support import NvFuserOperatorSupport
import torch._prims as prims
from torch.fx.passes.graph_drawer import FxGraphDrawer
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.fx.passes.fuser_utils import fuse_by_partitions

from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch.testing._internal.jit_utils import JitTestCase

class TestFXGraphPasses(JitTestCase):

    def forward1(a, b, c):
        add = a + b
        add_1 = add + b
        add_2 = add_1 + c
        relu_1 = add_2.relu()
        add_3 = add_1 + add_2
        add_4 = add_1 + relu_1 + add_3
        relu_2 = add_4.relu()
        add_5 = relu_2 + add_4
        add_6 = add_5 + add_4
        return add_4, add_6

    def forward2(a, b, _):
        add = a + b
        add_1 = add + b
        relu_1 = add_1.relu()  # blocked by this
        add_3 = add_1 + relu_1
        add_4 = add_1 + add_3
        return add_4, add_1

    def forward3(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return add, add_1, add_2

    def forward4(a, b, c):
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return torch.where(add > 0, add_1, add_2)

    def forward5(a, b, c):
        # add should be fused right branch, as left branch is not supported
        add = a + 1

        # left branch
        relu = add.relu()
        # right branch
        add_1 = add + 2

        return relu, add_1

    def forward6(a, b, c):
        # add should have its own partition, as neither branchs are supported
        add = a + 1

        # left branch
        relu = add.relu()
        # right branch
        relu_1 = add.relu()

        return relu, relu_1

    def forward7(a, b, c):
        # both branches are supported, but add should be merged with right branch, as right branch is larger
        add = a + 1

        # left branch
        add_1 = add + 2

        # right branch is larger
        add_2 = add + 1
        add_3 = add_2 + 1

        return add_3, add_1

    def forward8(a, b, c):
        # both branches are in the same partition, add should join the same partition
        add = a + 1

        # left branch
        add_1 = add + 2

        # right branch
        add_2 = add + 1

        # left and right branch merges
        add_3 = add_2 + add_1

        return add_3

    def forward9(a, b, c):
        add = a + 1

        # branch 1
        add_1 = add + 1

        # branch 2
        add_2 = add + 1

        # branch_3
        add_3 = add + 1

        out = torch.stack([add_1, add_2, add_3])

        return out

    def forward10(a, b, c):
        add = a + 1

        # branch 1
        add_1 = add + 1

        # branch 2
        add_2 = add + 1

        # branch 3: depends on branch 2
        add_3 = add + add_2

        out = torch.stack([add_1, add_2, add_3])

        return out

    def forward11(a, b, c):
        add = a + 1

        # branch 1
        add_1 = add.relu()

        # branch 2 depends on branch 1
        add_2 = add + add_1

        # branch 3
        add_3 = add.relu()

        out = torch.stack([add_1, add_2, add_3])

        return out


    @parametrize("fn, expected_partition", [
        (forward1, [["add_7", "add_6"], ["add_5", "add_4", "add_3"], ["add_2", "add_1", "add"]]),
        (forward2, [["add_3", "add_2"], ["add_1", "add"]]),

        # 2 branches cases
        (forward5, [["add_1", "add"]]),
        (forward6, [["add"]]),
        (forward7, [["add_3", "add_2", "add"], ["add_1"]]),
        (forward8, [["add_3", "add_2", "add", "add_1"]]),

        # 3 branch cases
        (forward9, [['add_3'], ['add_2'], ['add_1', 'add']]),
        (forward10, [['add_3', 'add_2', 'add'], ['add_1']]),
        (forward11, [['add_1'], ['add']]),
    ])
    # failing cases
    # @parametrize("fn, expected_partition", [
    #     (forward3, [["add_2", "add_1", "add"]]),  # horizontal fusion without a common downstream node, not working yet
    #     (forward4, [["add_2", "add_1", "add"]]),  # horizontal fusion with a common downstream node, not working yet
    # ]
    def test_partitioner(self, fn, expected_partition):
        traced = symbolic_trace(fn)

        drawer = FxGraphDrawer(traced, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("before.png")

        class MockOperatorSupport(OperatorSupport):
            def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
                return node.op == "call_function" and node.target in {operator.add}

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced, supported_ops)

        candidates = partitioner.get_candidates()

        partitions = partitioner.partition(candidates)

        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]

        print("partitions_name", partitions_name)
        print("expected_partition", expected_partition)

        assert len(partitions_name) == len(expected_partition)
        for i in range(len(partitions_name)):
            assert set(partitions_name[i]) == set(expected_partition[i])

        fused_graph = partitioner.fuse_partitions(partitions)

        drawer = FxGraphDrawer(fused_graph, "test")
        dot_graph = drawer.get_dot_graph()
        dot_graph.write_png("after.png")

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c)
        result = fused_graph(a, b, c)

        torch.testing.assert_close(expected, result)


    def test_nvfuser_patition_real_models(self):
        # this test assumes torch_bench_graphs is place at torch's root folder
        test_cases = [
            "torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_forward_0",
            "torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_backward_0",
            "torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_backward_0",
            "torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_forward_0",
            "torch_bench_graphs/moco/moco_forward_4",
            "torch_bench_graphs/moco/moco_backward_0",
            "torch_bench_graphs/moco/moco_backward_7",
            "torch_bench_graphs/moco/moco_forward_9",
            "torch_bench_graphs/moco/moco_forward_3",
            "torch_bench_graphs/moco/moco_backward_10",
            "torch_bench_graphs/moco/moco_forward_7",
            "torch_bench_graphs/moco/moco_backward_9",
            "torch_bench_graphs/moco/moco_backward_3",
            "torch_bench_graphs/moco/moco_forward_10",
            "torch_bench_graphs/moco/moco_backward_4",
            "torch_bench_graphs/moco/moco_forward_0",
            "torch_bench_graphs/moco/moco_backward_6",
            "torch_bench_graphs/moco/moco_forward_5",
            "torch_bench_graphs/moco/moco_backward_2",
            "torch_bench_graphs/moco/moco_forward_2",
            "torch_bench_graphs/moco/moco_forward_8",
            "torch_bench_graphs/moco/moco_backward_11",
            "torch_bench_graphs/moco/moco_backward_1",
            "torch_bench_graphs/moco/moco_backward_5",
            "torch_bench_graphs/moco/moco_forward_1",
            "torch_bench_graphs/moco/moco_forward_6",
            "torch_bench_graphs/moco/moco_backward_8",
            "torch_bench_graphs/moco/moco_forward_11",
            "torch_bench_graphs/resnet18/resnet18_backward_0",
            "torch_bench_graphs/resnet18/resnet18_forward_0",
            "torch_bench_graphs/mnasnet1_0/mnasnet1_0_backward_0",
            "torch_bench_graphs/mnasnet1_0/mnasnet1_0_forward_0",
            "torch_bench_graphs/BERT_pytorch/BERT_pytorch_forward_0",
            "torch_bench_graphs/BERT_pytorch/BERT_pytorch_backward_0",
            "torch_bench_graphs/resnet50/resnet50_forward_0",
            "torch_bench_graphs/resnet50/resnet50_backward_0",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_0",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_1",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_0",
            "torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_1",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_1",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_3",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_2",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_0",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_2",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_0",
            "torch_bench_graphs/hf_Albert/hf_Albert_forward_1",
            "torch_bench_graphs/hf_Albert/hf_Albert_backward_3",
            "torch_bench_graphs/dlrm/dlrm_backward_0",
            "torch_bench_graphs/dlrm/dlrm_forward_0",
            "torch_bench_graphs/drq/drq_backward_0",
            "torch_bench_graphs/drq/drq_forward_1",
            "torch_bench_graphs/drq/drq_backward_1",
            "torch_bench_graphs/drq/drq_forward_0",
            "torch_bench_graphs/pytorch_struct/pytorch_struct_backward_0",
            "torch_bench_graphs/pytorch_struct/pytorch_struct_forward_0",
            "torch_bench_graphs/Background_Matting/Background_Matting_backward_0",
            "torch_bench_graphs/Background_Matting/Background_Matting_forward_0",
            "torch_bench_graphs/timm_regnet/timm_regnet_forward_0",
            "torch_bench_graphs/timm_regnet/timm_regnet_backward_0",
            "torch_bench_graphs/hf_Bert/hf_Bert_forward_1",
            "torch_bench_graphs/hf_Bert/hf_Bert_backward_1",
            "torch_bench_graphs/hf_Bert/hf_Bert_backward_2",
            "torch_bench_graphs/hf_Bert/hf_Bert_forward_2",
            "torch_bench_graphs/hf_Bert/hf_Bert_forward_0",
            "torch_bench_graphs/hf_Bert/hf_Bert_backward_0",
            "torch_bench_graphs/densenet121/densenet121_backward_0",
            "torch_bench_graphs/densenet121/densenet121_forward_0",
            "torch_bench_graphs/timm_nfnet/timm_nfnet_backward_0",
            "torch_bench_graphs/timm_nfnet/timm_nfnet_forward_0",
            "torch_bench_graphs/squeezenet1_1/squeezenet1_1_forward_0",
            "torch_bench_graphs/squeezenet1_1/squeezenet1_1_backward_0",
            "torch_bench_graphs/alexnet/alexnet_forward_0",
            "torch_bench_graphs/alexnet/alexnet_backward_0",
            "torch_bench_graphs/Super_SloMo/Super_SloMo_forward_0",
            "torch_bench_graphs/Super_SloMo/Super_SloMo_backward_0",
            "torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_backward_0",
            "torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_forward_0",
            "torch_bench_graphs/maml_omniglot/maml_omniglot_backward_0",
            "torch_bench_graphs/maml_omniglot/maml_omniglot_forward_0",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_1",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_13",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_0",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_7",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_6",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_11",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_9",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_3",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_10",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_2",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_8",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_12",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_5",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_4",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_6",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_10",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_7",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_12",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_0",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_1",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_4",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_13",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_5",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_2",
            "torch_bench_graphs/hf_Bart/hf_Bart_backward_8",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_9",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_3",
            "torch_bench_graphs/hf_Bart/hf_Bart_forward_11",
            "torch_bench_graphs/timm_resnest/timm_resnest_forward_0",
            "torch_bench_graphs/timm_resnest/timm_resnest_backward_0",
            "torch_bench_graphs/mobilenet_v2/mobilenet_v2_backward_0",
            "torch_bench_graphs/mobilenet_v2/mobilenet_v2_forward_0",
            "torch_bench_graphs/timm_efficientnet/timm_efficientnet_forward_0",
            "torch_bench_graphs/timm_efficientnet/timm_efficientnet_backward_0",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_1",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_1",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_0",
            "torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_0",
            "torch_bench_graphs/mobilenet_v2_quantized_qat/mobilenet_v2_quantized_qat_backward_0",
            "torch_bench_graphs/mobilenet_v2_quantized_qat/mobilenet_v2_quantized_qat_forward_0",
            "torch_bench_graphs/LearningToPaint/LearningToPaint_backward_0",
            "torch_bench_graphs/LearningToPaint/LearningToPaint_forward_0",
            "torch_bench_graphs/vgg16/vgg16_forward_0",
            "torch_bench_graphs/vgg16/vgg16_backward_0",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_1",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_6",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_1",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_6",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_11",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_8",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_2",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_5",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_8",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_2",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_5",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_10",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_7",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_0",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_7",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_0",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_4",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_11",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_3",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_9",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_4",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_forward_10",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_3",
            "torch_bench_graphs/hf_GPT2/hf_GPT2_backward_9",
            "torch_bench_graphs/pytorch_unet/pytorch_unet_backward_0",
            "torch_bench_graphs/pytorch_unet/pytorch_unet_forward_0",
            "torch_bench_graphs/dcgan/dcgan_backward_0",
            "torch_bench_graphs/dcgan/dcgan_forward_0",
            "torch_bench_graphs/timm_vovnet/timm_vovnet_forward_0",
            "torch_bench_graphs/timm_vovnet/timm_vovnet_backward_0",
            "torch_bench_graphs/hf_T5/hf_T5_forward_7",
            "torch_bench_graphs/hf_T5/hf_T5_forward_13",
            "torch_bench_graphs/hf_T5/hf_T5_backward_0",
            "torch_bench_graphs/hf_T5/hf_T5_backward_11",
            "torch_bench_graphs/hf_T5/hf_T5_backward_7",
            "torch_bench_graphs/hf_T5/hf_T5_forward_0",
            "torch_bench_graphs/hf_T5/hf_T5_forward_14",
            "torch_bench_graphs/hf_T5/hf_T5_backward_9",
            "torch_bench_graphs/hf_T5/hf_T5_backward_3",
            "torch_bench_graphs/hf_T5/hf_T5_forward_10",
            "torch_bench_graphs/hf_T5/hf_T5_forward_4",
            "torch_bench_graphs/hf_T5/hf_T5_backward_12",
            "torch_bench_graphs/hf_T5/hf_T5_forward_9",
            "torch_bench_graphs/hf_T5/hf_T5_forward_3",
            "torch_bench_graphs/hf_T5/hf_T5_backward_4",
            "torch_bench_graphs/hf_T5/hf_T5_backward_6",
            "torch_bench_graphs/hf_T5/hf_T5_forward_1",
            "torch_bench_graphs/hf_T5/hf_T5_backward_10",
            "torch_bench_graphs/hf_T5/hf_T5_forward_12",
            "torch_bench_graphs/hf_T5/hf_T5_forward_6",
            "torch_bench_graphs/hf_T5/hf_T5_backward_1",
            "torch_bench_graphs/hf_T5/hf_T5_forward_2",
            "torch_bench_graphs/hf_T5/hf_T5_forward_8",
            "torch_bench_graphs/hf_T5/hf_T5_backward_5",
            "torch_bench_graphs/hf_T5/hf_T5_backward_13",
            "torch_bench_graphs/hf_T5/hf_T5_backward_14",
            "torch_bench_graphs/hf_T5/hf_T5_backward_2",
            "torch_bench_graphs/hf_T5/hf_T5_backward_8",
            "torch_bench_graphs/hf_T5/hf_T5_forward_5",
            "torch_bench_graphs/hf_T5/hf_T5_forward_11",
            "torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_backward_0",
            "torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_forward_0",
        ]

        device = 'cuda'
        draw = False

        for dir in test_cases:
            path = dir.split('/')
            model_name = path[-1]
            module_path = '.'.join(path)
            input_data_path = f'{dir}/{model_name}.input'

            print(f"====== {model_name} ======")

            module = importlib.import_module(module_path)

            m = module.FxModule()
            traced = symbolic_trace(m)

            if draw:
                print("Drawing original graph...")
                drawer = FxGraphDrawer(traced, "test")
                dot_graph = drawer.get_dot_graph()
                dot_graph.write_png("before.png")

            supported_ops = NvFuserOperatorSupport()
            partitioner = CapabilityBasedPartitioner(traced, supported_ops)

            print("Collecting fusable nodes...")
            candidates = partitioner.get_candidates()

            print("Forming partitions...")
            partitions = partitioner.partition(candidates)

            print("Partitions formed:")
            for partition in partitions:
                print([node.name for node in partition.nodes])

            print("Fusing partitions...")
            fused_graph = partitioner.fuse_partitions(partitions)

            # compile the nvFuser submodel with torchscript jit
            # for node in fused_graph.graph.nodes:
            #     if "fused_" in node.name:
            #         module = getattr(fused_graph, node.name)
            #         setattr(fused_graph, node.name, torch.jit.script(module) )

            if draw:
                print("Drawing fused graph...")
                drawer = FxGraphDrawer(fused_graph, "test")
                dot_graph = drawer.get_dot_graph()
                dot_graph.write_png("after.png")

            try:
                print("Generating testing data...")
                with (open(input_data_path, 'rb')) as f:
                    inputs_meta = pickle.load(f)

                    inputs = []
                    for meta in inputs_meta:
                        type, shape, stride, dtype = meta

                        if dtype in {torch.int, torch.int32, torch.int64, torch.bool, torch.int, torch.uint8}:
                            input = torch.randint(0, 1, shape, dtype=dtype, device=device)
                        else:
                            input = torch.rand(shape, dtype=dtype, device=device)

                        inputs.append(input)

                m.to(device)
                fused_graph.to(device)

                print("Running original model...")
                expected = m(*inputs)

                print("Running fused model...")
                result = fused_graph(*inputs)

                torch.testing.assert_close(expected, result, equal_nan=True, rtol=1e-5, atol=1e-5)
                print("Passed!")

            except Exception as e:
                print(f"{model_name} failed!", e)


    def test_fuser_util(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.param = torch.nn.Parameter(torch.rand(4, 4))

            def forward(self, a, b, c):
                add = a + b

                linear_1 = self.linear(add)

                add_1 = add + c
                add_2 = add_1 + self.param
                add_3 = add_1 + linear_1
                add_4 = add_2 + add_3

                linear_2 = self.linear2(add_4)

                add_5 = linear_2 + add_4
                add_6 = add_5 + a
                relu = add_6.relu()

                return add_4, add_6, relu

        m = TestModule()
        traced = symbolic_trace(m)

        print(traced.graph)

        # TODO: support for arbitrate node order
        test_cases = [
            [['add', 'add_1'], ['add_5', 'add_6']],
            [['add', 'add_1', 'add_2']],  # vertical fusion
            [['add_2', 'add_3']],         # horizontal fusion
            [['add_3', 'add_4']],
            [['add_6', 'add_5']],     # arbitray node order
            [['add_4', 'add_1', 'add_3', 'add_2']],           # arbitray node order
            [['add_5', 'add_6'], ['add_1', 'add_2', 'add_3', 'add_4']],  # arbitray partition order
            [['add_5', 'linear2']],   # includes call_function + call_module node
            [['add_6', 'relu']],   # includes call_function + call_module node
            [['param', 'add_2']],   # includes get_attr + call_module nodes
            [['param', 'add_1', 'linear']],   # includes get_attr + call_function + call_module nodes
            [["add", "linear", "add_1", "param", "add_2", "add_3", "add_4", "linear2", "add_5", "add_6", "relu"]],  # full graph
        ]

        # expected failing cases
        x_test_cases = [
            [['add', 'add_1'], ['add_1', 'add_5', 'add_6']],  # add_1 exists in multiple partitions
            [['add', 'add_1', 'add_3']],    # invalid partition: circular dependency
            [['add_4', 'add_5']],    # invalid partition: circular dependency
            [['relu', 'add_5']],    # invalid partition: circular dependency
        ]

        # drawer = FxGraphDrawer(traced, "test")
        # dot_graph = drawer.get_dot_graph()
        # dot_graph.write_png("before.png")

        for id, test_case in enumerate(test_cases):
            gm = copy.deepcopy(traced)
            nodes = gm.graph.nodes
            nodes_by_name = {node.name : node for node in nodes}

            partitions = []
            for names in test_case:
                partitions.append([nodes_by_name[name] for name in names])

            fused_graph = fuse_by_partitions(gm, partitions)

            # drawer = FxGraphDrawer(fused_graph, "test")
            # dot_graph = drawer.get_dot_graph()
            # dot_graph.write_png(f"after_{id}.png")

            a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

            expected = m(a, b, c)
            result = fused_graph(a, b, c)

            torch.testing.assert_close(expected, result)

    def test_nvfuser_prim_operator_support(self):
        def _wrapper(a, b, broadcast_dimensions):
            a_bc = prims.broadcast_in_dim(a, b.shape, broadcast_dimensions)
            return prims.add(a_bc, b)

        traced = symbolic_trace(_wrapper)

        supported_ops = NvFuserOperatorSupport()
        for node in traced.graph.nodes:
            if node.op in CALLABLE_NODE_OPS:
                assert supported_ops.is_node_supported({}, node)

instantiate_parametrized_tests(TestFXGraphPasses)

if __name__ == "__main__":
    run_tests()
