import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
model.load_state_dict(torch.load("mnist_cnn.pt"))
model.eval()  # set the model to evaluation mode

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
test_dataset = datasets.MNIST('./data', train=False, transform=transform)
dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Classify the generated images
model.eval()  # ensure the model is in evaluation mode
total = 0
correct = 0

for i, (images, labels) in enumerate(dataloader):
    # Move images and labels to device
    images = images.to(device)
    labels = labels.to(device)
    
    # Classify the images
    outputs = model(images)
    
    # Get the predicted classes
    _, predicted = torch.max(outputs.data, 1)
    
    # Update the counters
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    
# Compute overall accuracy
accuracy = correct / total * 100
print(f"Accuracy: {accuracy:.2f}%")