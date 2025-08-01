cd tests
export VLLM_WORKER_MULTIPROC_METHOD=spawn
pytest -v -s basic_correctness/test_cumem.py
pytest -v -s basic_correctness/test_basic_correctness.py
pytest -v -s basic_correctness/test_cpu_offload.py
VLLM_TEST_ENABLE_ARTIFICIAL_PREEMPT=1 pytest -v -s basic_correctness/test_preemption.py


# Examples Test
cd exmaples
pip install tensorizer

python3 offline_inference/basic/generate.py --model facebook/opt-125m
python3 offline_inference/basic/generate.py --model meta-llama/Llama-2-13b-chat-hf --cpu-offload-gb 10
python3 offline_inference/basic/chat.py
python3 offline_inference/prefix_caching.py
python3 offline_inference/llm_engine_example.py
python3 offline_inference/audio_language.py --seed 0
python3 offline_inference/vision_language.py --seed 0
python3 offline_inference/vision_language_pooling.py --seed 0
python3 offline_inference/vision_language_multi_image.py --seed 0
VLLM_USE_V1=0 python3 others/tensorize_vllm_model.py --model facebook/opt-125m serialize --serialized-directory /tmp/ --suffix v1 && python3 others/tensorize_vllm_model.py --model facebook/opt-125m deserialize --path-to-tensors /tmp/vllm/facebook/opt-125m/v1/model.tensors
python3 offline_inference/encoder_decoder.py
python3 offline_inference/encoder_decoder_multimodal.py --model-type whisper --seed 0
python3 offline_inference/basic/classify.py
python3 offline_inference/basic/embed.py
python3 offline_inference/basic/score.py
VLLM_USE_V1=0 python3 offline_inference/profiling.py --model facebook/opt-125m run_num_steps --num-steps 2

# cuda Test

pytest -v -s cuda/test_cuda_context.py

# Basic Models Test
pytest -v -s models/test_transformers.py
pytest -v -s models/test_registry.py
pytest -v -s models/test_utils.py
pytest -v -s models/test_vision.py
pytest -v -s models/test_initialization.py


# Multi-Modal Models Test (Standard)
pip install git+https://github.com/TIGER-AI-Lab/Mantis.git
pytest -v -s models/multimodal/processing
pytest -v -s --ignore models/multimodal/generation/test_whisper.py models/multimodal -m core_model
cd .. && pytest -v -s tests/models/multimodal/generation/test_whisper.py -m core_model  # Otherwise, mp_method="spawn" doesn't work

#Quantized Models Test
pytest -v -s models/quantization
