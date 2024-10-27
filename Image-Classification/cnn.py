import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score
import numpy as np

torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Resize((64, 64)),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])



dataset = datasets.ImageFolder(root='./petimages', transform=transform)


test_size = int(0.2 * len(dataset))
train_size = len(dataset) - test_size
test_set, train_set = torch.utils.data.random_split(dataset, [train_size, test_size])


trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False, num_workers=2)


learning_rate = 0.001
batch_size = 32
epoch_size = 3  


# CNN Model
class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Simplified classifier
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256),  
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 16 * 16)  
        x = self.classifier(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu' 
cnn = CNN().to(device) 

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)


# Model Training
cnn.train() # turn on train mode, this is a good practice to do
for epoch in range(epoch_size):  

    loss = 0.0 
    print(f'Epoch {epoch+1}/{epoch_size}')  

    for i, data in enumerate(trainloader, 0):
        # get the inputs and label from dataloader
        inputs, labels = data
        # move tensors to your current device (cpu or gpu)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients using zero_grad()
        optimizer.zero_grad()

        
        outputs = cnn(inputs)
        loss2 = criterion(outputs, labels)
        loss2.backward()
        optimizer.step()

        # print some statistics
        loss += loss2.item() 
        if i % 50 == 49:    
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss / 50:.3f}')
            loss = 0.0

print('Finished Training')


# evaluation on evaluation set
ground_truth = []
prediction = []
cnn.eval() 
with torch.no_grad(): 
    for data in testloader:
        inputs, labels = data
        inputs = inputs.to(device)
        ground_truth.extend(labels.numpy()) 
        # calculate outputs by running inputs through the network
        outputs = cnn(inputs)
        _, predicted = torch.max(outputs.data, 1)
        prediction.extend(predicted.cpu().numpy())


accuracy = accuracy_score(ground_truth, prediction)
recall = recall_score(ground_truth, prediction, average='macro')
precision = precision_score(ground_truth, prediction, average='macro')