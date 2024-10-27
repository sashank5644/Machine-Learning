import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(42)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(1 , 1) 

    def forward(self, x):

        out = self.fc(x) 
        
        return out 



net = Net()


optimizer = optim.SGD(net.parameters(), lr=0.01) 


X = 2 * torch.rand(100) - 1 
Y = X * 3.0 + 1


# start training (10 epochs) with pytorch
for epoch in range(10): 
    epoch_loss = 0 
    for i, (x, y) in enumerate(zip(X, Y)):
        
        
        x = torch.unsqueeze(x, 0) 
        y = torch.unsqueeze(y, 0) 

        optimizer.zero_grad()
        
        output = net(x) 
        
        loss = nn.MSELoss()(output, y) 

        # Back Propogation
        loss.backward()
        
        # update parameters with one optimization step
        optimizer.step()
        
        # add up loss for current example to the total loss for current epoch
        epoch_loss += loss.item()
        
    # print out some statistics
    print("Epoch {} - loss: {}".format(epoch + 1, epoch_loss))

 
for name, param in net.named_parameters():
    if "weight" in name:
        w = round(param.item(), 1)
    if "bias" in name:
        b = round(param.item(), 1)


