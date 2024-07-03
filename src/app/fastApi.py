import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import os


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


app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "../../model/mnist-0.0.1.pt")

input_size = 28 * 28
n_kernels = 6
output_size = 10

model = ConvNet(input_size, n_kernels, output_size)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])


@app.post("/api/v1/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True).item()

    return {"prediction": prediction}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
