import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import torch.optim as optim
import matplotlib.pyplot as plt


# Cellule 2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(datasets.MNIST("../../../data/raw", download=True, train=True, transform=tf),
                                           batch_size=64, shuffle=True)
test_load = torch.utils.data.DataLoader(datasets.MNIST("../../../data/raw", download=True, train=False, transform=tf),
                                        batch_size=64, shuffle=True)


batch = next(iter(train_loader))
x = batch[0][:10]
y = batch[1][:10]

n_kernels = 6
output_size = 10
input_size = 28 * 28


class ConvNet(nn.Module):
    def __init__(self, input_size, n_kernels, output_size):
        super(ConvNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=n_kernels, out_channels=n_kernels, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(n_kernels * 4 * 4, 50),
            nn.Linear(50, output_size)
        )

    def forward(self, x):
        return self.net(x)


model = ConvNet(input_size, n_kernels, output_size)

print(model)


class MLP(nn.Module):
    def __init__(self, input_size, n_hidden, output_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, output_size)
        )

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, optimizer, device, perm=torch.arange(0, 784).long(), n_epoch=1):
    model.train()
    for epoch in range(n_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if isinstance(model, MLP):
                data = data.view(-1, 28 * 28)
            else:

                data = data.view(-1, 28 * 28)
                data = data[:, perm]
                data = data.view(-1, 1, 28, 28)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(
                    f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')


model = ConvNet(input_size, n_kernels, output_size).to(device)
optimizer = optim.AdamW(model.parameters())

train(model, train_loader, optimizer, device, n_epoch=1)


def test(model, test_load, device, perm=torch.arange(0, 784).long()):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_load:
            data, target = data.to(device), target.to(device)

            if isinstance(model, MLP):
                data = data.view(-1, 28 * 28)
            else:
                data = data.view(-1, 28 * 28)
                data = data[:, perm]
                data = data.view(-1, 1, 28, 28)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_load.dataset)
    accuracy = 100. * correct / len(test_load.dataset)

    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_load.dataset)} ({accuracy:.2f}%)\n')


model = ConvNet(input_size, n_kernels, output_size).to(device)
optimizer = optim.AdamW(model.parameters())

train(model, train_loader, optimizer, device, n_epoch=1)
test(model, test_load, device)



def main():

    input_size = 28 * 28
    output_size = 10
    n_hidden = 8
    n_kernels = 6

    perm = torch.randperm(784)

    convnet = ConvNet(input_size, n_kernels, output_size)
    convnet.to(device)
    print("CNN")
    print(f"Parameters={sum(p.numel() for p in convnet.parameters())/1e3}K")

    optimizer = optim.AdamW(convnet.parameters())

    train(convnet, train_loader, optimizer, device, perm, n_epoch=1)
    test(convnet, test_load, device, perm)

    #MLP
    mlp = MLP(input_size, n_hidden, output_size)
    mlp.to(device)
    print("MLP")
    print(f"Parameters={sum(p.numel() for p in mlp.parameters()) / 1e3}K")

    optimizer = optim.AdamW(mlp.parameters())

    train(mlp, train_loader, optimizer, device, perm, n_epoch=1)
    test(mlp, test_load, device, perm)

    torch.save(convnet.state_dict(), "../../convnet_model.pt")
    print("CNN model saved to convnet_model.pt")


if __name__ == '__main__':
    main()


