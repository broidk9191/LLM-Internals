import torch
import torch.nn as nn

x= torch.randn(100,1)
y= 3*x + 2 + 0.1*torch.randn(100,1)

model = nn.Linear(1,1) #Linear Model
loss_fn = nn.MSELoss() #Mean Squared Error for regression
opt = torch.optim.SGD(model.parameters(), lr=0.1) #Stochastic Gradient Descent

for step in range(300):  #zeroing grad -> forward -> loss calc -> backward -> optimizing -> zero grad...
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred,y)
    loss.backward()
    opt.step()

print(model.weight.item(), model.bias.item())