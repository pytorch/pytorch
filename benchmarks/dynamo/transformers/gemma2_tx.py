from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

messages = [
    {"role": "user", "content": "Write me a poem about Machine Learning."},
]
input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

with torch.inference_mode():
    outputs = model.generate(**input_ids, max_new_tokens=256)

t0 = time.perf_counter()
with torch.inference_mode():
    outputs = model.generate(**input_ids, max_new_tokens=256)
t1 = time.perf_counter()

print(tokenizer.decode(outputs[0]))
print("Time :", t1 - t0)




#     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]

# t0 = time.perf_counter()
# with torch.inference_mode():
#     generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     generation = generation[0][input_len:]
# t1 = time.perf_counter()
# print("Time :", t1 - t0)
# decoded = processor.decode(generation, skip_special_tokens=True)
# print(decoded)

# # input_text = "Write me a poem about Machine Learning."
# # input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# # outputs = model.generate(**input_ids, max_new_tokens=32)
# # print(tokenizer.decode(outputs[0]))
