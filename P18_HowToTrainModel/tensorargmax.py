import torch

ouput = torch.tensor([[0.5,0.6],
                      [0.2,0.5]])

print(ouput.argmax(1))