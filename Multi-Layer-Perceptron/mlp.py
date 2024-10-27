import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(42)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.fc1 =  nn.Linear(1, 10)
        self.fc2 =  nn.Linear(10, 1)

    def forward(self, x):
        out =  self.fc1(x)
        out =  F.relu(out)
        out =  self.fc2(out)

        return out


net = Net()

optimizer = optim.SGD(net.parameters(), lr = 0.05) 

X = 2 * torch.rand(1000) - 1 
Y = X ** 2

# train 10 epochs
for epoch in range(10): 
    epoch_loss = 0 
    for i, (x, y) in enumerate(zip(X, Y)):

        x = torch.unsqueeze(x, 0) 
        y = torch.unsqueeze(y, 0) 

        optimizer.zero_grad()

        output = net(x) 

        loss = nn.MSELoss()(output, y) 

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print("Epoch {} - loss: {}".format(epoch + 1, epoch_loss))


X_test = torch.tensor([0.5, 0.3, 0.2])
Y_test = net(X_test.unsqueeze(1))
Y_test = Y_test.flatten().tolist()
Y_test = [round(y, 2) for y in Y_test]
