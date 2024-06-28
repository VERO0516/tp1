import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tf= transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../../data", download=True, train=True, transform=tf),
    batch_size=64, shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../../data", download=True, train=False, transform=tf),
    batch_size=64, shuffle=True
)


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


def train(model, train_loader, optimizer, device, n_epoch=1):
    model.train()
    for epoch in range(n_epoch):
        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Step: {batch_idx}, Train loss: {loss.item():.4f}')

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # 累积批次损失

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()  # 累积正确预测数

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')


def main():
    input_size = 28 * 28
    n_kernels = 6
    output_size = 10

    convnet = ConvNet(input_size, n_kernels, output_size)
    convnet.to(device)

    optimizer = optim.AdamW(convnet.parameters())

    train(convnet, train_loader, optimizer, device, n_epoch=5)
    test(convnet, test_loader, device)

    torch.save(convnet.state_dict(), "../../model/mnist-0.0.1.pt")
    print("CNN model saved to model/mnist-0.0.1.pt")


if __name__ == '__main__':
    main()
