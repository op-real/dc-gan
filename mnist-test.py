import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import natsort
import os
from PIL import Image

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

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

# Transformations applied to the images
transform = transforms.Compose([
    transforms.Grayscale(),  # ensure images are grayscale
    transforms.Resize((28, 28)),  # resize images to 28x28
    transforms.ToTensor(),  # convert to tensor
])
my_dataset = CustomDataSet("./gen_images", transform=transform)
dataloader = DataLoader(my_dataset , batch_size=10, shuffle=False, drop_last=False)

# Classify the generated images
model.eval()  # ensure the model is in evaluation mode

bin_count = torch.zeros(10).to(device)
conf = 0.0

for i, images in enumerate(dataloader):
    # Move images to device
    images = images.to(device)
    
    # Classify the images
    outputs = model(images)
    
    # Get the predicted classes
    confidence, predicted = torch.max(outputs.data, 1)
    
    _bin_count = torch.bincount(predicted, minlength=10)
    bin_count = bin_count + _bin_count
    conf = conf + torch.sum(confidence)
    
    # Print the predicted classes
print(f"Bin count: {bin_count}")
print(f"confidence: {conf}")
