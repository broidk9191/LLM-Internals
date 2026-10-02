import torch
import torch.nn as nn

x= torch.randn(100,1)
y= 3*x + 2 + 0.1*torch.randn(100,1)

model = nn.Linear(1,1)
loss_fn = nn.MSELoss()
opt = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(300):
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred,y)
    loss.backward()
    opt.step()

print(model.weight.item(), model.bias.item())