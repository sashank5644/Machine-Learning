import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(42)


class RNN(nn.Module):
    def __init__(self, input_feature_size, hidden_size, num_layers, num_classes, sequence_length):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        
        self.rnn = nn.RNN(input_size=input_feature_size, 
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         batch_first=True)

        
        self.fc = nn.Linear(sequence_length * hidden_size, num_classes)
    
    def forward(self, x):  
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        out, h_last = self.rnn(x, h0)
        
        out = out.reshape(out.size(0), -1)
        
        
        out = self.fc(out)

        return out  


# Test the RNN
x = torch.randn(10, 5, 512)  

# Set model parameters based on input tensor x
input_feature_size = 512  
hidden_size = 1024     
num_layers = 2         
num_classes = 3        
sequence_length = 5    

# Create model and test output shape
model = RNN(input_feature_size, hidden_size, num_layers, num_classes, sequence_length)
out_shape = list(model(x).shape)
print(out_shape)

# Print shape of the feed forward layer weights
for name, param in model.named_parameters():
    if name == "fc.weight":
        fc_shape = list(param.shape)
print(fc_shape)