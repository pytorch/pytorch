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


x = torch.tensor(120, dtype=torch.int8)
torchResult = mySum16(x)
dynamoResult = torch.compile(mySum16)(x)

assert(torchResult == dynamoResult == -16)

def mySum32(x):
    return (x + x).to(torch.int32)

x = torch.tensor( 35000, dtype=torch.int32)
torchResult = mySum32(x)
dynamoResult = torch.compile(mySum32)(x)

assert(torchResult == dynamoResult )
def mySum64(x):
    return (x+x).to(torch.int64)

x = torch.tensor( (2147483647), dtype=torch.int32)
torchResult = mySum64(x)
dynamoResult = torch.compile(mySum64)(x)

print(torchResult)
print(dynamoResult)

assert(torchResult == dynamoResult)