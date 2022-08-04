import copy
import logging

import torch
import torch.fx as fx
from functorch import make_fx
from functorch._src.remat_utils_mincut import rematerialize_stat, get_fused_graph, is_fused_node
from torch.profiler import profile, ProfilerActivity
from functorch._src.compile_utils import strip_overloads, fx_graph_cse
from torch.nn.utils.stateless import functional_call
from functorch.compile import ts_compile
from functorch.compile import default_decompositions
import torch.utils._pytree as pytree
from functorch.experimental import functionalize
from torchdynamo.testing import reduce_to_scalar_loss

USE_FUNCTIONALIZATION = True

test_cases_joint = ['torch_bench_graphs/hf_Bert/hf_Bert_joint_1',
 'torch_bench_graphs/hf_Bert/hf_Bert_joint_2',
 'torch_bench_graphs/hf_Bert/hf_Bert_joint_0',
 'torch_bench_graphs/resnet50/resnet50_joint_0',
 'torch_bench_graphs/vgg16/vgg16_joint_0',
 'torch_bench_graphs/squeezenet1_1/squeezenet1_1_joint_0',
 'torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_joint_0',
 'torch_bench_graphs/Background_Matting/Background_Matting_joint_0',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_6',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_7',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_1',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_10',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_9',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_11',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_2',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_0',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_4',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_8',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_5',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_joint_3',
 'torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_joint_0',
 'torch_bench_graphs/hf_DistilBert/hf_DistilBert_joint_0',
 'torch_bench_graphs/hf_DistilBert/hf_DistilBert_joint_1',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_joint_1',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_joint_0',
#  'torch_bench_graphs/hf_Longformer/hf_Longformer_joint_3', RecursionError in fusion
 'torch_bench_graphs/hf_Longformer/hf_Longformer_joint_2',
 'torch_bench_graphs/maml_omniglot/maml_omniglot_joint_0',
 'torch_bench_graphs/hf_Albert/hf_Albert_joint_1',
 'torch_bench_graphs/hf_Albert/hf_Albert_joint_3',
 'torch_bench_graphs/hf_Albert/hf_Albert_joint_0',
 'torch_bench_graphs/hf_Albert/hf_Albert_joint_2',
 'torch_bench_graphs/timm_nfnet/timm_nfnet_joint_0',
 'torch_bench_graphs/timm_regnet/timm_regnet_joint_0',
 'torch_bench_graphs/LearningToPaint/LearningToPaint_joint_0',
 'torch_bench_graphs/pytorch_CycleGAN_and_pix2pix/pytorch_CycleGAN_and_pix2pix_joint_0',
 'torch_bench_graphs/mobilenet_v3_large/mobilenet_v3_large_joint_0',
 'torch_bench_graphs/densenet121/densenet121_joint_0',
 'torch_bench_graphs/mobilenet_v2/mobilenet_v2_joint_0',
 'torch_bench_graphs/pytorch_struct/pytorch_struct_joint_0',
 'torch_bench_graphs/drq/drq_joint_0',
 'torch_bench_graphs/drq/drq_joint_1',
 'torch_bench_graphs/pytorch_stargan/pytorch_stargan_joint_0',
 'torch_bench_graphs/resnet18/resnet18_joint_0',
 'torch_bench_graphs/dcgan/dcgan_joint_0',
 'torch_bench_graphs/alexnet/alexnet_joint_0',
 'torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_joint_0',
 'torch_bench_graphs/timm_efficientdet/timm_efficientdet_joint_0',
 'torch_bench_graphs/soft_actor_critic/soft_actor_critic_joint_1',
 'torch_bench_graphs/soft_actor_critic/soft_actor_critic_joint_0',
 'torch_bench_graphs/BERT_pytorch/BERT_pytorch_joint_0',
 'torch_bench_graphs/Super_SloMo/Super_SloMo_joint_0',
 'torch_bench_graphs/timm_vovnet/timm_vovnet_joint_0',
 'torch_bench_graphs/timm_resnest/timm_resnest_joint_0',
 'torch_bench_graphs/attention_is_all_you_need_pytorch/attention_is_all_you_need_pytorch_joint_0',
 'torch_bench_graphs/timm_efficientnet/timm_efficientnet_joint_0',
 'torch_bench_graphs/pytorch_unet/pytorch_unet_joint_0',
 'torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_joint_0',
 'torch_bench_graphs/hf_T5/hf_T5_joint_11',
 'torch_bench_graphs/hf_T5/hf_T5_joint_8',
 'torch_bench_graphs/hf_T5/hf_T5_joint_12',
 'torch_bench_graphs/hf_T5/hf_T5_joint_6',
 'torch_bench_graphs/hf_T5/hf_T5_joint_2',
 'torch_bench_graphs/hf_T5/hf_T5_joint_3',
 'torch_bench_graphs/hf_T5/hf_T5_joint_0',
 'torch_bench_graphs/hf_T5/hf_T5_joint_10',
 'torch_bench_graphs/hf_T5/hf_T5_joint_1',
 'torch_bench_graphs/hf_T5/hf_T5_joint_5',
 'torch_bench_graphs/hf_T5/hf_T5_joint_13',
 'torch_bench_graphs/hf_T5/hf_T5_joint_4',
 'torch_bench_graphs/hf_T5/hf_T5_joint_14',
 'torch_bench_graphs/hf_T5/hf_T5_joint_9',
 'torch_bench_graphs/hf_T5/hf_T5_joint_7',
 'torch_bench_graphs/yolov3/yolov3_joint_1',
 'torch_bench_graphs/yolov3/yolov3_joint_3',
 'torch_bench_graphs/yolov3/yolov3_joint_2',
 'torch_bench_graphs/yolov3/yolov3_joint_4',
 'torch_bench_graphs/yolov3/yolov3_joint_0',
 'torch_bench_graphs/hf_Bart/hf_Bart_joint_1',
 'torch_bench_graphs/hf_Bart/hf_Bart_joint_0',
 'torch_bench_graphs/mnasnet1_0/mnasnet1_0_joint_0',
 'torch_bench_graphs/hf_Bert/hf_Bert_backward_1',
 'torch_bench_graphs/hf_Bert/hf_Bert_forward_2',
 'torch_bench_graphs/hf_Bert/hf_Bert_backward_2',
 'torch_bench_graphs/hf_Bert/hf_Bert_forward_1',
 'torch_bench_graphs/hf_Bert/hf_Bert_backward_0',
 'torch_bench_graphs/hf_Bert/hf_Bert_forward_0',
 'torch_bench_graphs/resnet50/resnet50_forward_0',
 'torch_bench_graphs/resnet50/resnet50_backward_0',
 'torch_bench_graphs/vgg16/vgg16_backward_0',
 'torch_bench_graphs/vgg16/vgg16_forward_0',
 'torch_bench_graphs/squeezenet1_1/squeezenet1_1_forward_0',
 'torch_bench_graphs/squeezenet1_1/squeezenet1_1_backward_0',
 'torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_backward_0',
 'torch_bench_graphs/timm_vision_transformer/timm_vision_transformer_forward_0',
 'torch_bench_graphs/Background_Matting/Background_Matting_backward_0',
 'torch_bench_graphs/Background_Matting/Background_Matting_forward_0',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_10',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_0',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_11',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_1',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_7',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_3',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_9',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_10',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_8',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_7',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_8',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_2',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_1',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_4',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_2',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_11',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_9',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_6',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_4',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_6',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_3',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_5',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_backward_5',
 'torch_bench_graphs/hf_GPT2/hf_GPT2_forward_0',
 'torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_backward_0',
 'torch_bench_graphs/resnext50_32x4d/resnext50_32x4d_forward_0',
 'torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_0',
 'torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_0',
 'torch_bench_graphs/hf_DistilBert/hf_DistilBert_backward_1',
 'torch_bench_graphs/hf_DistilBert/hf_DistilBert_forward_1',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_forward_2',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_forward_0',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_forward_3',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_backward_1',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_backward_0',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_backward_3',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_backward_2',
 'torch_bench_graphs/hf_Longformer/hf_Longformer_forward_1',
 'torch_bench_graphs/maml_omniglot/maml_omniglot_backward_0',
 'torch_bench_graphs/maml_omniglot/maml_omniglot_forward_0',
 'torch_bench_graphs/hf_Albert/hf_Albert_backward_1',
 'torch_bench_graphs/hf_Albert/hf_Albert_backward_3',
 'torch_bench_graphs/hf_Albert/hf_Albert_forward_1',
 'torch_bench_graphs/hf_Albert/hf_Albert_backward_2',
 'torch_bench_graphs/hf_Albert/hf_Albert_forward_3',
 'torch_bench_graphs/hf_Albert/hf_Albert_forward_0',
 'torch_bench_graphs/hf_Albert/hf_Albert_backward_0',
 'torch_bench_graphs/hf_Albert/hf_Albert_forward_2',
 'torch_bench_graphs/timm_nfnet/timm_nfnet_forward_0',
 'torch_bench_graphs/timm_nfnet/timm_nfnet_backward_0',
 'torch_bench_graphs/timm_regnet/timm_regnet_backward_0',
 'torch_bench_graphs/timm_regnet/timm_regnet_forward_0',
 'torch_bench_graphs/LearningToPaint/LearningToPaint_backward_0',
 'torch_bench_graphs/LearningToPaint/LearningToPaint_forward_0',
 'torch_bench_graphs/pytorch_CycleGAN_and_pix2pix/pytorch_CycleGAN_and_pix2pix_forward_0',
 'torch_bench_graphs/pytorch_CycleGAN_and_pix2pix/pytorch_CycleGAN_and_pix2pix_backward_0',
 'torch_bench_graphs/mobilenet_v3_large/mobilenet_v3_large_forward_0',
 'torch_bench_graphs/mobilenet_v3_large/mobilenet_v3_large_backward_0',
 'torch_bench_graphs/densenet121/densenet121_backward_0',
 'torch_bench_graphs/densenet121/densenet121_forward_0',
 'torch_bench_graphs/mobilenet_v2/mobilenet_v2_forward_0',
 'torch_bench_graphs/mobilenet_v2/mobilenet_v2_backward_0',
 'torch_bench_graphs/pytorch_struct/pytorch_struct_backward_0',
 'torch_bench_graphs/pytorch_struct/pytorch_struct_forward_0',
 'torch_bench_graphs/drq/drq_backward_1',
 'torch_bench_graphs/drq/drq_forward_1',
 'torch_bench_graphs/drq/drq_forward_0',
 'torch_bench_graphs/drq/drq_backward_0',
 'torch_bench_graphs/pytorch_stargan/pytorch_stargan_forward_0',
 'torch_bench_graphs/pytorch_stargan/pytorch_stargan_backward_0',
 'torch_bench_graphs/resnet18/resnet18_backward_0',
 'torch_bench_graphs/resnet18/resnet18_forward_0',
 'torch_bench_graphs/dcgan/dcgan_forward_0',
 'torch_bench_graphs/dcgan/dcgan_backward_0',
 'torch_bench_graphs/alexnet/alexnet_forward_0',
 'torch_bench_graphs/alexnet/alexnet_backward_0',
 'torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_forward_0',
 'torch_bench_graphs/nvidia_deeprecommender/nvidia_deeprecommender_backward_0',
 'torch_bench_graphs/timm_efficientdet/timm_efficientdet_backward_0',
 'torch_bench_graphs/timm_efficientdet/timm_efficientdet_forward_0',
 'torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_0',
 'torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_0',
 'torch_bench_graphs/soft_actor_critic/soft_actor_critic_forward_1',
 'torch_bench_graphs/soft_actor_critic/soft_actor_critic_backward_1',
 'torch_bench_graphs/BERT_pytorch/BERT_pytorch_forward_0',
 'torch_bench_graphs/BERT_pytorch/BERT_pytorch_backward_0',
 'torch_bench_graphs/Super_SloMo/Super_SloMo_backward_0',
 'torch_bench_graphs/Super_SloMo/Super_SloMo_forward_0',
 'torch_bench_graphs/timm_vovnet/timm_vovnet_forward_0',
 'torch_bench_graphs/timm_vovnet/timm_vovnet_backward_0',
 'torch_bench_graphs/timm_resnest/timm_resnest_backward_0',
 'torch_bench_graphs/timm_resnest/timm_resnest_forward_0',
 'torch_bench_graphs/attention_is_all_you_need_pytorch/attention_is_all_you_need_pytorch_backward_0',
 'torch_bench_graphs/attention_is_all_you_need_pytorch/attention_is_all_you_need_pytorch_forward_0',
 'torch_bench_graphs/timm_efficientnet/timm_efficientnet_backward_0',
 'torch_bench_graphs/timm_efficientnet/timm_efficientnet_forward_0',
 'torch_bench_graphs/pytorch_unet/pytorch_unet_forward_0',
 'torch_bench_graphs/pytorch_unet/pytorch_unet_backward_0',
 'torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_backward_0',
 'torch_bench_graphs/shufflenet_v2_x1_0/shufflenet_v2_x1_0_forward_0',
 'torch_bench_graphs/hf_T5/hf_T5_forward_11',
 'torch_bench_graphs/hf_T5/hf_T5_backward_9',
 'torch_bench_graphs/hf_T5/hf_T5_forward_4',
 'torch_bench_graphs/hf_T5/hf_T5_forward_14',
 'torch_bench_graphs/hf_T5/hf_T5_forward_13',
 'torch_bench_graphs/hf_T5/hf_T5_forward_3',
 'torch_bench_graphs/hf_T5/hf_T5_forward_10',
 'torch_bench_graphs/hf_T5/hf_T5_backward_12',
 'torch_bench_graphs/hf_T5/hf_T5_backward_7',
 'torch_bench_graphs/hf_T5/hf_T5_backward_4',
 'torch_bench_graphs/hf_T5/hf_T5_backward_10',
 'torch_bench_graphs/hf_T5/hf_T5_forward_1',
 'torch_bench_graphs/hf_T5/hf_T5_backward_0',
 'torch_bench_graphs/hf_T5/hf_T5_backward_6',
 'torch_bench_graphs/hf_T5/hf_T5_backward_8',
 'torch_bench_graphs/hf_T5/hf_T5_forward_12',
 'torch_bench_graphs/hf_T5/hf_T5_forward_7',
 'torch_bench_graphs/hf_T5/hf_T5_backward_1',
 'torch_bench_graphs/hf_T5/hf_T5_forward_2',
 'torch_bench_graphs/hf_T5/hf_T5_backward_3',
 'torch_bench_graphs/hf_T5/hf_T5_forward_6',
 'torch_bench_graphs/hf_T5/hf_T5_forward_0',
 'torch_bench_graphs/hf_T5/hf_T5_backward_2',
 'torch_bench_graphs/hf_T5/hf_T5_backward_5',
 'torch_bench_graphs/hf_T5/hf_T5_backward_11',
 'torch_bench_graphs/hf_T5/hf_T5_forward_9',
 'torch_bench_graphs/hf_T5/hf_T5_backward_14',
 'torch_bench_graphs/hf_T5/hf_T5_forward_5',
 'torch_bench_graphs/hf_T5/hf_T5_forward_8',
 'torch_bench_graphs/hf_T5/hf_T5_backward_13',
 'torch_bench_graphs/yolov3/yolov3_forward_3',
 'torch_bench_graphs/yolov3/yolov3_forward_2',
 'torch_bench_graphs/yolov3/yolov3_backward_4',
 'torch_bench_graphs/yolov3/yolov3_forward_4',
 'torch_bench_graphs/yolov3/yolov3_backward_2',
 'torch_bench_graphs/yolov3/yolov3_backward_3',
 'torch_bench_graphs/yolov3/yolov3_forward_0',
 'torch_bench_graphs/yolov3/yolov3_forward_1',
 'torch_bench_graphs/yolov3/yolov3_backward_0',
 'torch_bench_graphs/yolov3/yolov3_backward_1',
 'torch_bench_graphs/hf_Bart/hf_Bart_backward_0',
 'torch_bench_graphs/hf_Bart/hf_Bart_backward_1',
 'torch_bench_graphs/hf_Bart/hf_Bart_forward_1',
 'torch_bench_graphs/hf_Bart/hf_Bart_forward_0',
 'torch_bench_graphs/mnasnet1_0/mnasnet1_0_forward_0',
 'torch_bench_graphs/mnasnet1_0/mnasnet1_0_backward_0']


non_zero_mincut_memory_graphs_functionalization = set([
'timm_nfnet_joint_0',
'timm_nfnet_backward_0',
'BERT_pytorch_joint_0',
'hf_Albert_joint_2',
'BERT_pytorch_backward_0',
'hf_T5_joint_11',
'hf_T5_joint_8',
'hf_T5_joint_12',
'hf_T5_joint_10',
'hf_T5_joint_13',
'hf_T5_joint_9',
'hf_T5_joint_0',
'hf_T5_joint_2',
'hf_T5_joint_3',
'hf_T5_joint_1',
'hf_T5_joint_5',
'hf_T5_joint_4',
'hf_T5_backward_9',
'hf_T5_backward_12',
'hf_T5_backward_10',
'hf_T5_backward_8',
'hf_T5_backward_11',
'hf_T5_backward_13',
'attention_is_all_you_need_pytorch_joint_0',
'Super_SloMo_joint_0',
'timm_efficientdet_joint_0',
'hf_T5_joint_6',
'hf_T5_joint_14',
'hf_T5_backward_4',
'hf_T5_backward_0',
'hf_T5_backward_6',
'hf_T5_backward_1',
'hf_T5_backward_3',
'hf_T5_backward_2',
'hf_T5_backward_5',
'hf_T5_backward_14',
'hf_T5_forward_0',
'nvidia_deeprecommender_joint_0',
'hf_Bart_joint_1',
'hf_Bart_joint_0',
'yolov3_joint_2',
'yolov3_joint_4',
'mobilenet_v3_large_joint_0',
'yolov3_joint_1',
'pytorch_struct_joint_0',
'yolov3_joint_3'
])

# SKIP_CASES = set(zero_fusion_group).union(set(one_fusion_group))
SKIP_CASES = {}#set(zero_fusion_group_functionalization).union(set(one_fusion_group_functionalization))



def get_test_cases():
    return test_cases_joint


def get_skip_cases():
    return SKIP_CASES


def get_non_zero_mincut_memory_graphs():
    if USE_FUNCTIONALIZATION:
        return non_zero_mincut_memory_graphs_functionalization
    else:
        return non_zero_mincut_memory_graphs
    


def strip_overloads_save(gm):
    """
    Modifies the target of graph nodes in :attr:`gm` to strip overloads.

    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
    overload_dict = {}
    for node in gm.graph.nodes:
        if isinstance(node.target, torch._ops.OpOverload):
            overload_dict[node.name] = node.target
            node.target = node.target.overloadpacket
    gm.recompile()
    return overload_dict


def get_cuda_time(timing):
    """
    Get the total cuda time from torch profiler timings
    """
    cuda_time_total = 0
    for e in timing:
        cuda_time_total = cuda_time_total + e.cuda_time_total
    return cuda_time_total


def benchmark_GPU_time(f, inp, list_inp, itr = 5):
    """
    Return the average CUDA time of an iteration of ``f`` on inputs ``inp``
    Using `with torch.no_grad`.

    Args:
        f: The function to profile
        inp: The input
        list_inp(bool): if True, profile f(*inp), otherwise f(inp)
        itr(int): The number of iterations to run, default to 5
    """
    if list_inp:
        with torch.no_grad():
            for _ in range(5):
                f(*inp)
                torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(itr):
                    f(*inp)
                    torch.cuda.synchronize()
        return get_cuda_time(prof.key_averages()) / itr

    with torch.no_grad():
        for _ in range(5):
            f(inp)
            torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(itr):
                f(inp)
                torch.cuda.synchronize()

        # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
        return get_cuda_time(prof.key_averages()) / itr


def profile_scripted_graph(traced_graph, inp, list_inp, itr = 5):
    """
    Return the average cuda time of the jit.scripted version of `traced_graph` on input `inp`

    Args:
        traced_graph(fx.GraphModule): The graph to profile
        inp: The input
        list_inp(bool): if True, profile f(*inp), otherwise f(inp)
        itr(int): The number of iterations to run, default to 5
    """
    traced_graph.graph.eliminate_dead_code()
    traced_graph.recompile()
    script_f = ts_compile(traced_graph, inp)
    avg_cuda_time_f = benchmark_GPU_time(script_f, inp, list_inp, itr = itr)
    return avg_cuda_time_f


def get_num_fused_group(gm):
    num_fusion_group = 0
    for node in gm.graph.nodes:
        if is_fused_node(node):
            num_fusion_group += 1
    return num_fusion_group


def profile_fused_graph(fused_graph, inp, list_inp, overload_dict = None, itr = 5):
    """
    Return the average cuda time of the jit.scripted version of `fused_graph` on input `inp`, 
    and the number of fusion groups in `fused_graph`
    Speficically, each fused group of `fused_graph` is scriptedif there is at least one fusion group.
    Otherwise, the whole graph is scripted.

    Args:
        fused_graph(fx.GraphModule): The graph to profile
        inp: The input
        list_inp(bool): if True, profile f(*inp), otherwise f(inp)
        itr(int): The number of iterations to run, default to 5
        overload_dict(Dict[torch._ops.OpOverloadPacket -> torch._ops.OpOverload]): If not None, all
                overloadPacket-op nodes' target in fused_graph will be replaced by overload ops
                in `overload_dict`, which is keyed by node.name.
    """
    num_fusion_group = 0
    for node in fused_graph.graph.nodes:
        if is_fused_node(node):
            module = getattr(fused_graph, node.name)
            setattr(fused_graph, node.name, ts_compile(module, 0) )
            num_fusion_group += 1
        elif isinstance(node.target, torch._ops.OpOverloadPacket) and overload_dict is not None:
            if node.name in overload_dict:
                node.target = overload_dict[node.name]

    fused_graph.recompile()
    if num_fusion_group == 0: # no fused group
        script_f = ts_compile(fused_graph, 0)#torch.jit.script(fused_graph)
        return benchmark_GPU_time(script_f, inp, list_inp, itr = itr), 0

    avg_cuda_time_g = benchmark_GPU_time(fused_graph, inp, list_inp, itr = itr)
    return avg_cuda_time_g, num_fusion_group


def profile_graph(name, traced_graph, inp, eager_inp=False):
    # Profile eager time
    traced_graph_copy = copy.deepcopy(traced_graph)
    eager_time = benchmark_GPU_time(traced_graph, inp, eager_inp) # can't strup overloads here

    # Profile jit.scripted time
    traced_graph.recompile()
    inp, spec  = pytree.tree_flatten(inp)
    avg_cuda_time_f = profile_scripted_graph(traced_graph, inp, True)

    # CSE pass
    traced_graph = traced_graph_copy 
    # strip_overloads(traced_graph)
    # traced_graph.recompile()
    csed = fx_graph_cse(traced_graph.graph)
    csed_graph =  fx.GraphModule(traced_graph, csed)
    overload_dict = strip_overloads_save(csed_graph)
    csed_graph.recompile()
    csed_graph_copy = copy.deepcopy(csed_graph)
    
    # Profile fused graph time
    fused_graph = get_fused_graph(csed_graph)
    avg_cuda_time_g, num_fusion_group = profile_fused_graph(fused_graph, inp, True, overload_dict = overload_dict)

    # Profile rematerialized graph time
    stat = {}
    fused_graph = rematerialize_stat(csed_graph_copy, stat)
    num_remat_group = stat["num_group_remat"]
    memory_reduced = stat["memory_reduced"]
    num_node_pairs = stat["num_node_pairs"]
    avg_cuda_time_h, _ = profile_fused_graph(fused_graph, inp, True, overload_dict = overload_dict)

    print(f"{name}, {eager_time}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fusion_group}, {num_remat_group}, {memory_reduced}, {num_node_pairs}", flush=True)



def profile_module(name, m, inp):
    def fake_fn(args):
        return m(*args)
    
    if USE_FUNCTIONALIZATION:
        logging.info("Using functionalization")
        traced_graph = make_fx(functionalize(fake_fn, remove='mutations_and_views'), decomposition_table=default_decompositions)(inp)
    else:
        logging.info("Not using functionalization")
        traced_graph = make_fx(fake_fn, decomposition_table=default_decompositions)(inp)
    traced_graph.graph.set_codegen(torch.fx.graph.CodeGen())  # avoid recursive pytree
    profile_graph(name, traced_graph, inp)


def trace_model(model, inputs):
    """
    Get the full graph (both forward and backward) of `model` on `inputs`
    The moddel should have a single forward and a single backward graph
    """
    def f(params, inp):
        out = functional_call(model, params, inp)
        loss = reduce_to_scalar_loss(out)
        loss.backward()
        return [param.grad for param in params.values()]
    
    params = dict(model.named_parameters())
    traced_graph = make_fx(f, decomposition_table=default_decompositions)(params, inputs)
    return traced_graph, params


def trace_model_functionalization(model, inputs):
    """
    Get the full graph (both forward and backward) of `model` on `inputs`
    The moddel should have a single forward and a single backward graph
    """
    def f(params, inp):
        out = functional_call(model, params, inp)
        loss = reduce_to_scalar_loss(out)
        loss.backward()
        return [param.grad for param in params.values()]
    
    params = dict(model.named_parameters())
    traced_graph = make_fx(f, decomposition_table=default_decompositions)(params, inputs)
    def fake_fn(params, inputs):
        return traced_graph(params, inputs)
    fx_g = make_fx(functionalize(fake_fn))(params, inputs)
    return fx_g, params


def profile_model(name, model, inputs):
    """
    Profile a model on inputs
    """
    traced_graph, params = trace_model(model, inputs)
    traced_graph.graph.set_codegen(torch.fx.graph.CodeGen())  # avoid recursive pytree
    traced_graph_copy = copy.deepcopy(traced_graph)
    
    eager_time = benchmark_GPU_time(traced_graph, (params, inputs), True) # can't strip overloads here

    arg_list, spec  = pytree.tree_flatten([params, inputs])
    script_f = ts_compile(traced_graph, 0)
    avg_cuda_time_f = benchmark_GPU_time(script_f, arg_list, True)# profile_scripted_graph(traced_graph, inp, True)

    traced_graph = traced_graph_copy
    traced_graph.graph.set_codegen(torch.fx.graph.CodeGen())  # avoid recursive pytree
    csed = fx_graph_cse(traced_graph.graph)
    csed_graph =  fx.GraphModule(traced_graph, csed)
    overload_dict = strip_overloads_save(csed_graph)
    csed_graph.recompile()
    csed_graph_copy = copy.deepcopy(csed_graph)
    
    fused_graph = get_fused_graph(csed_graph)
    avg_cuda_time_g, num_fusion_group = profile_fused_graph(fused_graph, arg_list, True, overload_dict = overload_dict)

    stat = {}
    fused_graph = rematerialize_stat(csed_graph_copy, stat)
    num_remat_group = stat["num_group_remat"]
    memory_reduced = stat["memory_reduced"]
    num_node_pairs = stat["num_node_pairs"]
    avg_cuda_time_h, _ = profile_fused_graph(fused_graph, arg_list, True, overload_dict = overload_dict)

    print(f"{name}, {eager_time}, {avg_cuda_time_f}, {avg_cuda_time_g}, {avg_cuda_time_h}, {num_fusion_group}, {num_remat_group}, {memory_reduced}, {num_node_pairs}", flush=True)




def check_remat_info(name, traced_graph, inputs):
    """
    Print the information about rematerialization on `traced_graph` (fx.Graph)
    """
    strip_overloads(traced_graph)
    traced_graph.recompile()

    csed = fx_graph_cse(traced_graph.graph)
    csed_graph =  fx.GraphModule(traced_graph, csed)

    stat = {}
    fused_graph = rematerialize_stat(csed_graph, stat)
    num_remat_group = stat["num_group_remat"]
    memory_reduced = stat["memory_reduced"]
    num_node_pairs = stat["num_node_pairs"]
    num_fusion_group = get_num_fused_group(fused_graph)
    print(f" '{name}',  {num_fusion_group}, {num_remat_group}, {memory_reduced}, {num_node_pairs}", flush=True)


def check_remat_info_gm(name, gm, inputs):
    """
    Print the information about rematerialization on `gm`(fx.GraphModule)
    """
    def fake_fn(args):
        return gm(*args)
    
    if USE_FUNCTIONALIZATION:
        traced_graph = make_fx(functionalize(fake_fn, remove='mutations_and_views'), decomposition_table=default_decompositions)(inputs)
    else:
        traced_graph = make_fx(fake_fn, decomposition_table=default_decompositions)(inputs)
    check_remat_info(name, traced_graph, inputs)


def check_remat_info_model(name, model, inputs):
    """
    Print the information about rematerialization on `model` (torchbench models)
    The moddel should have a single forward and a single backward graph
    """
    # if USE_FUNCTIONALIZATION:
    #     traced_graph, params = trace_model_functionalization(model, inputs)
    # else:
    traced_graph, params = trace_model(model, inputs)
    check_remat_info(name, traced_graph, inputs)