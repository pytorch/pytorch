# Import generic wrappers
from transformers import AutoModel, AutoTokenizer 

# Define the model repo
model_name = "sshleifer/tiny-gpt2" 

# Download pytorch model
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Transform input tokens 
inputs = tokenizer("Hello world!", return_tensors="pt")

# Model apply
outputs = model(**inputs)

# Export tiny GPT2.
from torch.onnx._internal._fx import export
#onnx_model = export(model, **inputs)
input_ids = inputs["input_ids"]
onnx_model = export(model, input_ids)
#attention_mask = inputs["attention_mask"]
#onnx_model = export(model, input_ids, attention_mask=attention_mask)