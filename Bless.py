import torch 
from torch import nn 

labels_map = {
    "0": "Angry",
    "1": "Disgust", 
    "2": "Fear",
    "3": "Happy", 
    "4": "Sad", 
    "5": "Suprise",
    "6": "Neutral"
}

class Bless1(nn.Module): 
  def __init__(self): 
    super(Bless1, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
    self.relu2 = nn.ReLU()
    self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
    self.dropout1 = nn.Dropout(0.25)

    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
    self.relu3 = nn.ReLU()
    self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
    self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
    self.relu4 = nn.ReLU()
    self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2))
    self.dropout2 = nn.Dropout(0.25)

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc1 = nn.LazyLinear(1024)
    self.relu5 = nn.ReLU()
    self.dropout3 = nn.Dropout(0.5)
    self.fc2 = nn.Linear(1024, len(labels_map))

  def _forward_impl(self, x: torch.Tensor) -> torch.Tensor: 
    x = self.conv1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.relu2(x)
    x = self.maxpool1(x)
    x = self.dropout1(x)

    x = self.conv3(x)
    x = self.relu3(x)
    x = self.maxpool2(x)
    x = self.conv4(x)
    x = self.relu4(x)
    x = self.maxpool3(x)
    x = self.dropout2(x)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc1(x)
    x = self.relu5(x)
    x = self.dropout3(x)
    x = self.fc2(x)
    return x 

  def forward(self, x): 
    return self._forward_impl(x)