set -x
set -e

for enable in 0 1; do
    for model in "AlbertForMaskedLM" "BartForCausalLM" "BartForConditionalGeneration" "BlenderbotSmallForCausalLM" "BlenderbotSmallForConditionalGeneration" "DebertaV2ForQuestionAnswering" "ElectraForCausalLM" "M2M100ForConditionalGeneration" "MBartForCausalLM" "MBartForConditionalGeneration" "OPTForCausalLM" "PLBartForCausalLM" "PLBartForConditionalGeneration" "PegasusForCausalLM" "Speech2Text2ForCausalLM" "TrOCRForCausalLM" "XGLMForCausalLM"; do
        TORCHINDUCTOR_NEW_CONFIGS=$enable TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python benchmarks/dynamo/huggingface.py --performance --inference --bfloat16 --backend inductor --device cuda --only $model
    done
done


for enable in 0 1; do
    for model in "BERT_pytorch" "LearningToPaint" "alexnet" "dcgan" "demucs" "densenet121" "dlrm" "fastNLP_Bert" "mobilenet_v2" "phlippe_densenet" "phlippe_resnet" "pytorch_stargan" "resnet18" "shufflenet_v2_x1_0" "speech_transformer" "squeezenet1_1" "stable_diffusion_text_encoder" "timm_efficientdet" "timm_nfnet" "timm_resnest" "timm_vision_transformer" "timm_vovnet" "vgg16" "hf_T5"; do
        TORCHINDUCTOR_NEW_CONFIGS=$enable TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 TORCHINDUCTOR_MAX_AUTOTUNE=1 python benchmarks/dynamo/torchbench.py --performance --inference --bfloat16 --backend inductor --device cuda --only $model
    done
done
