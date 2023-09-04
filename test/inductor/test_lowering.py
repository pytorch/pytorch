import torch
import torch._dynamo
import torch._dynamo.config

def mySum16(x):
    return (x + x).to(torch.int16)
def myMul16(x):
    return (x * x).to(torch.int16)
def mySquare16(x):
    return (x ** 2).to(torch.int16)

x = torch.tensor(128, dtype=torch.uint8)

torchResult = mySum16(x)
dynamoResult = torch.compile(mySum16)(x)

assert(torchResult == dynamoResult == 0)

torchResult = myMul16(x)
dynamoResult = torch.compile(myMul16)(x)

assert(torchResult == dynamoResult == 0)

torchResult = mySquare16(x)
dynamoResult = torch.compile(mySquare16)(x)

assert(torchResult == dynamoResult == 0)