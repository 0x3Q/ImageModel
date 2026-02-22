import os
import sys
import torch
import torchvision
import torchmetrics

DATASET_DIRECTORY = "images"
DATASET_PATCH_SIZE = 256
DATASET_PATCH_PIXELS = DATASET_PATCH_SIZE * DATASET_PATCH_SIZE
DATASET_PATCH_SCALING = 1.0 / 255.0
MODEL_EPOCHS = 100000
MODEL_BATCH_SIZE = 16
MODEL_LEARNING_RATE = 1e-4
MODEL_TESTING_INTERVAL = 10
MODEL_LAGRANGE_MULTIPLIER = 1e-2
MODEL_GRADIENT_CLIPPING_THRESHOLD = 1.0

class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, x, minimum):
        context.save_for_backward(x, minimum)
        return torch.max(x, minimum)

    @staticmethod
    def backward(context, gradient):
        x, minimum = context.saved_tensors
        return gradient * ((x >= minimum) | (gradient < 0.0)).float(), None

class NegativeLogarithmFunction:
    @staticmethod
    def apply(x, epsilon):
        return -LowerBoundFunction.apply(x, epsilon).log2()

class LowerBoundParameter(torch.nn.Module):
    def __init__(
        self,
        value,
        minimum = 0.0,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon ** 2))
        self.value = torch.nn.Parameter(torch.sqrt(value + self.epsilon))
        self.register_buffer("minimum", torch.tensor((minimum + epsilon ** 2) ** 0.5))

    def forward(self):
        return LowerBoundFunction.apply(self.value, self.minimum) ** 2 - self.epsilon

class GDN(torch.nn.Module):
    def __init__(
        self,
        channels,
        inverse = False,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.inverse = inverse
        self.channels = channels
        self.beta = LowerBoundParameter(torch.ones(channels), minimum=minimum, epsilon=epsilon)
        self.gamma = LowerBoundParameter(origin * torch.eye(channels), minimum=0.0, epsilon=epsilon)

    def forward(self, x):
        beta = self.beta()
        gamma = self.gamma().view(self.channels, self.channels, 1, 1)
        output = torch.nn.functional.conv2d(x ** 2, gamma, beta)
        if self.inverse:
            return x * torch.sqrt(output)
        else:
            return x * torch.rsqrt(output)

class Encoder(torch.nn.Module):
    def __init__(
        self,
        features,
        channels,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(features, channels, kernel_size=5, stride=2, padding=2)
        self.activation1 = GDN(channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation2 = GDN(channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.convolution3 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation3 = GDN(channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.convolution4 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = self.convolution3(x)
        x = self.activation3(x)
        x = self.convolution4(x)
        return x

class HyperEncoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.activation1 = torch.nn.ReLU()
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation2 = torch.nn.ReLU()
        self.convolution3 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = self.convolution3(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(
        self,
        channels,
        features,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.convolution1 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation1 = GDN(channels, inverse=True, minimum=minimum, origin=origin, epsilon=epsilon)
        self.convolution2 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation2 = GDN(channels, inverse=True, minimum=minimum, origin=origin, epsilon=epsilon)
        self.convolution3 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation3 = GDN(channels, inverse=True, minimum=minimum, origin=origin, epsilon=epsilon)
        self.convolution4 = torch.nn.ConvTranspose2d(channels, features, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = self.convolution3(x)
        x = self.activation3(x)
        x = self.convolution4(x)
        return x

class HyperDecoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convolution1 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation1 = torch.nn.ReLU()
        self.convolution2 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation2 = torch.nn.ReLU()
        self.convolution3 = torch.nn.ConvTranspose2d(channels, channels * 2, kernel_size=3, stride=1, padding=1, output_padding=0)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = self.convolution3(x)
        return x

class Quantizer(torch.nn.Module):
    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).uniform_(-0.5, 0.5)
        else:
            return x.round()

class RateLoss(torch.nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))

    def forward(self, x, means, scales):
        upper = self.get_gaussian_cumulative(x + 0.5, means, scales)
        lower = self.get_gaussian_cumulative(x - 0.5, means, scales)
        return NegativeLogarithmFunction.apply(upper - lower, self.epsilon).sum(dim=[1, 2, 3])

    def get_gaussian_cumulative(self, x, means, scales):
        return torch.special.ndtr((x - means) / LowerBoundFunction.apply(scales, self.epsilon))

class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        features = 3,
        channels = 192,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.encoder = Encoder(features, channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.hyper_encoder = HyperEncoder(channels)
        self.quantizer = Quantizer()
        self.hyper_decoder = HyperDecoder(channels)
        self.decoder = Decoder(channels, features, minimum=minimum, origin=origin, epsilon=epsilon)
        self.rate_loss = RateLoss(epsilon=epsilon)
        self.hyper_means = torch.nn.Parameter(torch.zeros([channels, 1, 1]))
        self.hyper_scales = torch.nn.Parameter(torch.ones([channels, 1, 1]))

    def forward(self, x):
        x = self.encoder(x)
        y = self.hyper_encoder(x)
        x = self.quantizer(x)
        y = self.quantizer(y)
        y_bits = self.rate_loss(y, self.hyper_means, self.hyper_scales)
        means, scales = self.hyper_decoder(y).chunk(2, dim=1)
        x_bits = self.rate_loss(x, means, scales)
        x = self.decoder(x)
        return x, x_bits + y_bits

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory, patch_size):
        self.images = []
        self.random_crop = torchvision.transforms.RandomCrop(patch_size)
        self.random_flip = torchvision.transforms.RandomHorizontalFlip()
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.images.append(
                        torchvision.io.read_image(os.path.join(root, file))
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.random_flip(self.random_crop(self.images[index]))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    bpp_metric = torchmetrics.MeanMetric().to(device)
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), MODEL_LEARNING_RATE)
    dataset = Dataset(DATASET_DIRECTORY, patch_size=DATASET_PATCH_SIZE)
    testing_dataset, training_dataset = torch.utils.data.random_split(dataset, [0.2, 0.8])
    testing_batches = torch.utils.data.DataLoader(testing_dataset, batch_size=MODEL_BATCH_SIZE, shuffle=False)
    training_batches = torch.utils.data.DataLoader(training_dataset, batch_size=MODEL_BATCH_SIZE, shuffle=True)

    for epoch in range(1, MODEL_EPOCHS + 1):
        model.train()
        bpp_metric.reset()
        psnr_metric.reset()
        for samples in training_batches:
            samples = samples.to(device).float() * DATASET_PATCH_SCALING
            outputs, bits = model(samples)
            rate_loss = bits.mean() / DATASET_PATCH_PIXELS
            distortion_loss = torch.nn.functional.mse_loss(outputs, samples) * DATASET_PATCH_PIXELS
            (rate_loss + distortion_loss * MODEL_LAGRANGE_MULTIPLIER).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MODEL_GRADIENT_CLIPPING_THRESHOLD)
            optimizer.step()
            optimizer.zero_grad()
            bpp_metric.update(bits / DATASET_PATCH_PIXELS)
            psnr_metric.update(outputs, samples)
        print(f"[EPOCH {epoch}]: BPP: {bpp_metric.compute().item():.5f} PSNR: {psnr_metric.compute().item():.3f}")
        if epoch % MODEL_TESTING_INTERVAL == 0:
            model.eval()
            bpp_metric.reset()
            psnr_metric.reset()
            with torch.no_grad():
                for samples in testing_batches:
                    samples = samples.to(device).float() * DATASET_PATCH_SCALING
                    outputs, bits = model(samples)
                    bpp_metric.update(bits / DATASET_PATCH_PIXELS)
                    psnr_metric.update(outputs, samples)
                print(f"[EPOCH {epoch} - TESTING]: BPP: {bpp_metric.compute().item():.5f} PSNR: {psnr_metric.compute().item():.3f}")
