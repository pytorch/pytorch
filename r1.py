import torch
import torchvision
from PIL import Image
import requests

with torch.no_grad():
   model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
   model.eval()

   torch._dynamo.reset()

   compiled = torch.compile(model, dynamic=True, backend="inductor")

   url = "http://images.cocodataset.org/val2017/000000039769.jpg"
   image = Image.open(requests.get(url, stream=True).raw)
   image_tensor = torchvision.transforms.functional.to_tensor(image).unsqueeze(0)
   print(compiled(image_tensor))
