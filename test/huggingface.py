from transformers import BertModel, BertTokenizer, BertConfig
import torch

# for _ in range(5):
#     torch.matmul(torch.rand(1024, 1024), torch.rand(1024, 1024))
# assert False
enc = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizing input text
text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
tokenized_text = enc.tokenize(text)

# Masking one of the input tokens
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# Creating a dummy input
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
dummy_input = [tokens_tensor, segments_tensors]

# Initializing the model with the torchscript flag
# Flag set to True even though it is not necessary as this model does not have an LM Head.
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
    num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

# Instantiating the model
model = BertModel(config)

# The model needs to be in evaluation mode
model.eval()
torch.set_num_threads(1)
inp = [tokens_tensor, segments_tensors]

# If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors]).eval()
frozen_model = torch.jit.freeze(traced_model)
torch._C._jit_pass_convert_frozen_ops_to_mkldnn(frozen_model.graph)
traced_model = torch.jit.freeze(traced_model)
traced_model(*inp)
frozen_model(*inp)

NITER = 4
import time

# Time unfused execution
s = time.time()
for _ in range(NITER):
    traced_model(*inp)
e = time.time()
elapsed_unfused_sec = (e - s) / NITER


# Time fused execution
s = time.time()
for _ in range(NITER):
    frozen_model(*inp)
e = time.time()
elapsed_fused_sec = (e - s) / NITER

speedup_pct = (elapsed_unfused_sec - elapsed_fused_sec) / elapsed_unfused_sec * 100

print(speedup_pct)
print("Hugging face transformers", torch.get_num_threads(), elapsed_unfused_sec, elapsed_fused_sec, speedup_pct, sep=',')
# out = torch.jit.freeze(traced_model.eval())
# print(out.graph)
# import pdb; pdb.set_trace()
# torch.jit.save(traced_model, "traced_bert.pt")
#
