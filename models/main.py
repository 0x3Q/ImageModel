import os
import torch
import torchvision

class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, x, minimum):
        context.save_for_backward(x, minimum)
        return torch.max(x, minimum)

    @staticmethod
    def backward(context, gradient):
        x, minimum = context.saved_tensors
        return torch.where((x >= minimum) | (gradient < 0.0), gradient, 0.0), None

class LowerBoundParameter(torch.nn.Module):
    def __init__(
        self,
        value,
        minimum = 0.0,
        epsilon = 2 ** -18,
    ):
        super().__init__()
        self.register_buffer("minimum", torch.tensor((minimum + epsilon ** 2) ** 0.5))
        self.register_buffer("epsilon", torch.tensor(epsilon ** 2))
        self.value = torch.nn.Parameter(torch.sqrt(torch.max(value + self.epsilon, self.epsilon)))

    def forward(self):
        return LowerBoundFunction.apply(self.value, self.minimum) ** 2 - self.epsilon

class GDN(torch.nn.Module):
    def __init__(
        self,
        channels,
        inverse = False,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 2 ** -18,
    ):
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        self.biases = LowerBoundParameter(torch.ones(channels), minimum=minimum, epsilon=epsilon)
        self.weights = LowerBoundParameter(origin * torch.eye(channels), epsilon=epsilon)

    def forward(self, x):
        biases = self.biases()
        weights = self.weights().view(self.channels, self.channels, 1, 1)
        output = torch.nn.functional.conv2d(x ** 2, weights, biases)
        if self.inverse:
            return x * torch.sqrt(output)
        else:
            return x * torch.rsqrt(output)

class Encoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = GDN(channels, inverse=False)
        self.convolution1 = torch.nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2)
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.convolution3 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.convolution4 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation(x)
        x = self.convolution2(x)
        x = self.activation(x)
        x = self.convolution3(x)
        x = self.activation(x)
        x = self.convolution4(x)
        return x

class HyperEncoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.convolution1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.convolution3 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation(x)
        x = self.convolution2(x)
        x = self.activation(x)
        x = self.convolution3(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = GDN(channels, inverse=True)
        self.deconvolution1 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconvolution2 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconvolution3 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconvolution4 = torch.nn.ConvTranspose2d(channels, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.deconvolution1(x)
        x = self.activation(x)
        x = self.deconvolution2(x)
        x = self.activation(x)
        x = self.deconvolution3(x)
        x = self.activation(x)
        x = self.deconvolution4(x)
        return x

class HyperDecoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = torch.nn.LeakyReLU()
        self.deconvolution1 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconvolution2 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconvolution3 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.deconvolution1(x)
        x = self.activation(x)
        x = self.deconvolution2(x)
        x = self.activation(x)
        x = self.deconvolution3(x)
        return x

class Quantization(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.activation = torch.nn.Hardtanh(-2.0, 2.0)

    def forward(self, x):
        x = self.activation(x)
        if self.training:
            return x + torch.empty_like(x).uniform_(-0.5, 0.5)
        else:
            return x.round()

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(192)
        self.hyper_encoder = HyperEncoder(192)
        self.decoder = Decoder(192 * 2)
        self.hyper_decoder = HyperDecoder(192)
        self.quantization = Quantization()

    def forward(self, x):
        x = self.encoder(x)
        y = self.hyper_encoder(x)
        x = self.quantization(x)
        y = self.quantization(y)
        y = self.hyper_decoder(y)
        x = self.decoder(torch.cat([x, y], dim=1))
        return x

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.images = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", "jpeg")):
                    self.images.append(os.path.join(root, file))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torchvision.io.decode_image(self.images[index], mode="RGB").float() / 255.0

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = ImageDataset("images")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    criterion = torch.nn.MSELoss()

    epochs = 10000
    for epoch in range(1, epochs + 1):
        model.train()
        total_error = 0.0
        for images in dataloader:
            images = images.to(device)
            optimizer.zero_grad()
            output = model(images)
            error = criterion(output, images)
            error.backward()
            optimizer.step()
            total_error += error.detach().item() * images.size(0)
        average_error = total_error / len(dataset)
        print(f"[Epoch {epoch}/{epochs}]: Error: {average_error:.6f}")