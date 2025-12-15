import os
import sys
import torch
import torchvision
import torchmetrics

DATASET_DIRECTORY="images"
MODEL_TRAINING_EPOCHS = 100000
MODEL_TRAINING_BATCH_SIZE = 32
MODEL_TRAINING_LEARNING_RATE = 1e-4
MODEL_TESTING_BATCH_SIZE = 32
MODEL_TESTING_EPOCHS_INTERVAL = 10

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
        self.convolution1 = torch.nn.Conv2d(3, channels, kernel_size=5, stride=2, padding=2)
        self.activation1 = GDN(channels, inverse=False)
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation2 = GDN(channels, inverse=False)
        self.convolution3 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = self.convolution3(x)
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
        self.deconvolution1 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation1 = GDN(channels, inverse=True)
        self.deconvolution2 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation2 = GDN(channels, inverse=True)
        self.deconvolution3 = torch.nn.ConvTranspose2d(channels, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.deconvolution1(x)
        x = self.activation1(x)
        x = self.deconvolution2(x)
        x = self.activation2(x)
        x = self.deconvolution3(x)
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

class VectorQuantizer(torch.nn.Module):
    def __init__(self, features, symbols):
        super().__init__()
        self.symbols = torch.nn.Embedding(symbols, features)
        self.symbols.weight.data.uniform_(-1.0 / symbols, 1.0 / symbols)

    def forward(self, x):
        distances = self.get_squared_distances(x, self.symbols.weight.permute(1, 0))
        symbols = self.symbols(distances.argmin(dim=-1))
        deviation = torch.nn.functional.mse_loss(symbols, x.detach()) + torch.nn.functional.mse_loss(x, symbols.detach()) * 0.25
        return x + (symbols - x).detach(), deviation

    def get_squared_distances(self, x, y):
        return torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(y ** 2, dim=-2, keepdim=True) - torch.matmul(x, y) * 2.0

class ProductQuantizer(torch.nn.Module):
    def __init__(self, features, splits, symbols):
        super().__init__()
        self.quantizers = torch.nn.ModuleList(VectorQuantizer(features // splits, symbols) for _ in range(splits))

    def forward(self, x):
        outputs = self.get_outputs_from_quantizers(x, len(self.quantizers))
        errors = [error for _, error in outputs]
        symbols = [symbol for symbol, _ in outputs]
        return torch.cat(symbols, dim=-1), sum(errors)

    def get_outputs_from_quantizers(self, x, chunks):
        return [quantizer(x) for x, quantizer in zip(torch.chunk(x, chunks, dim=-1), self.quantizers)]

class Autoencoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(192)
        self.hyper_encoder = HyperEncoder(192)
        self.quantizer = ProductQuantizer(192, 6, 256)
        self.hyper_quantizer = ProductQuantizer(192, 6, 256)
        self.decoder = Decoder(192 * 2)
        self.hyper_decoder = HyperDecoder(192)

    def forward(self, x):
        x = self.encoder(x)
        y = self.hyper_encoder(x)
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        x, x_error = self.quantizer(x)
        y, y_error = self.hyper_quantizer(y)
        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)
        y = self.hyper_decoder(y)
        x = self.decoder(torch.cat([x, y], dim=1))
        return x, x_error + y_error

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.images = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", "jpeg")):
                    path = os.path.join(root, file)
                    self.images.append(
                        torchvision.io.decode_image(path, mode="RGB").float() / 255.0
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(DATASET_DIRECTORY)
    model = Autoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), MODEL_TRAINING_LEARNING_RATE)
    criterion = torch.nn.MSELoss()
    testing_dataset, training_dataset = torch.utils.data.random_split(dataset, [0.2, 0.8])
    testing_batches = torch.utils.data.DataLoader(testing_dataset, batch_size=MODEL_TESTING_BATCH_SIZE, shuffle=False)
    training_batches = torch.utils.data.DataLoader(training_dataset, batch_size=MODEL_TRAINING_BATCH_SIZE, shuffle=True)
    metrics = torchmetrics.image.PeakSignalNoiseRatio(1.0).to(device)

    for epoch in range(1, MODEL_TRAINING_EPOCHS + 1):
        model.train()
        metrics.reset()
        for samples in training_batches:
            samples = samples.to(device)
            outputs, error = model(samples)
            (error + criterion(outputs, samples)).backward()
            optimizer.step()
            optimizer.zero_grad()
            metrics.update(outputs, samples)
        print(f"[EPOCH {epoch}]: PSNR: {metrics.compute().item():.3f}")
        if epoch % MODEL_TESTING_EPOCHS_INTERVAL == 0:
            model.eval()
            metrics.reset()
            with torch.no_grad():
                for samples in testing_batches:
                    samples = samples.to(device)
                    outputs, _ = model(samples)
                    metrics.update(outputs, samples)
                print(f"[EPOCH {epoch} - TESTING]: PSNR: {metrics.compute().item():.3f}")
