import torch

x = torch.rand(2, 3)
y = torch.rand(3, 3)
z = [x, y]
z = torch.as_tensor(z, device="cuda")
print(z.shape)
