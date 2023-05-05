import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision import transforms

"""A simple implementation of LeNet model
Lecun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998).
"Gradient-based learning applied to document recognition".
Proceedings of the IEEE. 86 (11): 2278–2324.
"""
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=1,
                                out_channels=6,
                                kernel_size=(5,5),
                                stride=(1,1),
                                padding=(0,0))

        self.conv2 = nn.Conv2d(in_channels=6,
                                out_channels=16,
                                kernel_size=(5,5),
                                stride=(1,1),
                                padding=(0,0))

        self.conv3 = nn.Conv2d(in_channels=16,
                                out_channels=120,
                                kernel_size=(5,5),
                                stride=(1,1),
                                padding=(0,0))

        self.linear1 = nn.Linear(120, 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x

if __name__ == "__main__":

   # Load MNIST dataset
   train_set = datasets.FashionMNIST(
       root="data",
       train=True,
       download=True,
       transform = transforms.Compose([
       transforms.Resize((32,32)),
       transforms.ToTensor()]),
       target_transform=None
    )

   test_set = datasets.FashionMNIST(
       root="data",
       download=True,
       train=False,
        transform = transforms.Compose([
       transforms.Resize((32,32)),
       transforms.ToTensor()])
   )

   train_loader = DataLoader(train_set,
                            batch_size=32,
                            shuffle=True)

   test_loader = DataLoader(test_set,
                            batch_size=32,
                            shuffle=True)

   model = LeNet()
   loss = nn.CrossEntropyLoss()
   learning_rate = 0.001
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
   epochs = 5

   for epoch in range(epochs):
    for images, labels in train_loader:

        outputs = model(images)
        loss = loss(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
