helloworld1:
  g0 = True

  @torch.compile3
  def model(a, b, c):
    if g0:
      e = a * b + c
    else:
      e = b * c + a

    if e[0].item() >= 0:
      print(">= 0")
    else:
      print("< 0")

    return e + 1

  g0 = True
  model(a, b, c)
  model(a, b, c)
  g0 = False
  model(a, b, c)

--------------------------------------------------------------------------------
helloworld2:
  autograd
    "normal"
    forward+loss+backward+optimizer
  nn.module
    buffers
    parameters
