import os
import sys
import csv
import torch
import torchvision
import torchmetrics

DATASET_TESTING_DIRECTORY = "images"
DATASET_TRAINING_DIRECTORY = "..."
DATASET_PATCH_SIZE = 256
DATASET_PATCH_PIXELS = DATASET_PATCH_SIZE * DATASET_PATCH_SIZE
DATASET_PATCH_SCALING = 1.0 / 255.0
DATASET_PATCH_UNSCALING = 1.0 / DATASET_PATCH_SCALING ** 2

MODEL_FEATURES = 3
MODEL_CHANNELS = 192
MODEL_EXPERTS = 16
MODEL_CAPACITY = 1
MODEL_LAMBDA_VARIANT = 6
MODEL_LAMBDA_MULTIPLIER = 2 ** -MODEL_LAMBDA_VARIANT
MODEL_UNIQUE_VARIANT_STRING = f"VAEMOE{MODEL_EXPERTS}-V{MODEL_LAMBDA_VARIANT}"

SCRIPT_TRAINING_EPOCHS = 4000
SCRIPT_TRAINING_BATCH_SIZE = 16
SCRIPT_TRAINING_LEARNING_RATE = 1e-4
SCRIPT_TRAINING_GRADIENT_CLIPPING_THRESHOLD = 1.0
SCRIPT_TESTING_INTERVAL = 10
SCRIPT_TESTING_ITERATIONS = 16
SCRIPT_CHECKPOINT_IMPORT = None
SCRIPT_CHECKPOINT_SAVING_INTERVAL = 250

class LowerBoundFunction(torch.autograd.Function):
    @staticmethod
    def forward(context, x, minimum):
        context.save_for_backward(x, minimum)
        return torch.max(x, minimum)

    @staticmethod
    def backward(context, gradient):
        x, minimum = context.saved_tensors
        return gradient * ((x >= minimum) | (gradient < 0.0)), None

class NegativeLogarithmFunction:
    @staticmethod
    def apply(x, epsilon):
        return -torch.log2(LowerBoundFunction.apply(x, epsilon))

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
        gamma = self.gamma().reshape(self.channels, self.channels, 1, 1)
        output = torch.nn.functional.conv2d(x ** 2, gamma, beta)
        if self.inverse:
            return x * torch.sqrt(output)
        else:
            return x * torch.rsqrt(output)

class Experts(torch.nn.Module):
    def __init__(
        self,
        channels,
        experts,
        capacity,
    ):
        super().__init__()
        self.convolution1 = torch.nn.Conv1d(channels * experts, channels * experts * capacity, kernel_size=1, groups=experts)
        self.activation1 = torch.nn.SiLU()
        self.convolution2 = torch.nn.Conv1d(channels * experts * capacity, channels * experts, kernel_size=1, groups=experts)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = self.convolution2(x)
        return x

    def get_outputs_from_symbols(self, symbols):
        return self(symbols.flatten(1).unsqueeze(2)).view_as(symbols)

class ExpertsMixture(torch.nn.Module):
    def __init__(
        self,
        channels,
        experts = 16,
        capacity = 1,
        scaling = -0.5,
    ):
        super().__init__()
        self.scaling = torch.nn.Parameter(torch.tensor(channels ** scaling))
        self.experts = Experts(channels, experts, capacity)
        self.weights = torch.nn.Conv2d(channels, experts, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, symbols):
        weights = self.weights(symbols) * self.scaling
        symbols = torch.einsum("BCHW, BEHW -> BEC", symbols, self.get_softmax_from_weights(weights, 2, 3))
        symbols = self.experts.get_outputs_from_symbols(symbols)
        symbols = torch.einsum("BEC, BEHW -> BCHW", symbols, self.get_softmax_from_weights(weights, 1, 1))
        return symbols

    def get_softmax_from_weights(self, weights, start, finish):
        if start == finish:
            return torch.softmax(weights, dim=start)
        else:
            return torch.softmax(weights.flatten(start, finish), dim=start).view_as(weights)

class Encoder(torch.nn.Module):
    def __init__(
        self,
        features,
        channels,
        experts = 16,
        capacity = 1,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(features, channels, kernel_size=5, stride=2, padding=2)
        self.activation1 = GDN(channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.experts1 = ExpertsMixture(channels, experts=experts, capacity=capacity)
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation2 = GDN(channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.experts2 = ExpertsMixture(channels, experts=experts, capacity=capacity)
        self.convolution3 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation3 = GDN(channels, minimum=minimum, origin=origin, epsilon=epsilon)
        self.experts3 = ExpertsMixture(channels, experts=experts, capacity=capacity)
        self.convolution4 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = x + self.experts1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = x + self.experts2(x)
        x = self.convolution3(x)
        x = self.activation3(x)
        x = x + self.experts3(x)
        x = self.convolution4(x)
        return x

class HyperEncoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convolution1 = torch.nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.activation1 = torch.nn.LeakyReLU()
        self.convolution2 = torch.nn.Conv2d(channels, channels, kernel_size=5, stride=2, padding=2)
        self.activation2 = torch.nn.LeakyReLU()
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
        experts = 16,
        capacity = 1,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.convolution1 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation1 = GDN(channels, inverse=True, minimum=minimum, origin=origin, epsilon=epsilon)
        self.experts1 = ExpertsMixture(channels, experts=experts, capacity=capacity)
        self.convolution2 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation2 = GDN(channels, inverse=True, minimum=minimum, origin=origin, epsilon=epsilon)
        self.experts2 = ExpertsMixture(channels, experts=experts, capacity=capacity)
        self.convolution3 = torch.nn.ConvTranspose2d(channels, channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation3 = GDN(channels, inverse=True, minimum=minimum, origin=origin, epsilon=epsilon)
        self.experts3 = ExpertsMixture(channels, experts=experts, capacity=capacity)
        self.convolution4 = torch.nn.ConvTranspose2d(channels, features, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x):
        x = self.convolution1(x)
        x = self.activation1(x)
        x = x + self.experts1(x)
        x = self.convolution2(x)
        x = self.activation2(x)
        x = x + self.experts2(x)
        x = self.convolution3(x)
        x = self.activation3(x)
        x = x + self.experts3(x)
        x = self.convolution4(x)
        return x

class HyperDecoder(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.convolution1 = torch.nn.ConvTranspose2d(channels * 3 // 3, channels * 4 // 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation1 = torch.nn.LeakyReLU()
        self.convolution2 = torch.nn.ConvTranspose2d(channels * 4 // 3, channels * 5 // 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.activation2 = torch.nn.LeakyReLU()
        self.convolution3 = torch.nn.ConvTranspose2d(channels * 5 // 3, channels * 6 // 3, kernel_size=3, stride=1, padding=1, output_padding=0)

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

class GaussianLikelihood(torch.nn.Module):
    def __init__(
        self,
        minimum = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.register_buffer("minimum", torch.tensor(minimum))

    def forward(self, x, means, scales):
        upper = self.get_gaussian_cumulative(x, +0.5, means, scales)
        lower = self.get_gaussian_cumulative(x, -0.5, means, scales)
        return NegativeLogarithmFunction.apply(upper - lower, self.epsilon).sum(dim=[1, 2, 3])

    def get_gaussian_cumulative(self, x, offset, means, scales):
        return torch.special.ndtr((offset - torch.abs(x - means)) / LowerBoundFunction.apply(scales, self.minimum))

class FactorizedCumulative(torch.nn.Module):
    def __init__(
        self,
        channels,
        encoded = 3,
        decoded = 3,
        factored = True,
    ):
        super().__init__()
        self.channels = channels
        self.biases = torch.nn.Parameter(torch.empty(channels * decoded).uniform_(-0.5, 0.5))
        self.weights = torch.nn.Parameter(torch.empty(channels * decoded, encoded, 1, 1).fill_(-1.5))
        self.factored = factored
        if self.factored:
            self.factors = torch.nn.Parameter(torch.zeros(channels * decoded, 1, 1))

    def forward(self, x):
        weights = torch.nn.functional.softplus(self.weights)
        outputs = torch.nn.functional.conv2d(x, weights, self.biases, groups=self.channels)
        if self.factored:
            return outputs + torch.tanh(outputs) * torch.tanh(self.factors)
        else:
            return outputs

class FactorizedLikelihood(torch.nn.Module):
    def __init__(
        self,
        channels,
        features = (1, 3, 3, 3, 1),
        epsilon = 1e-6,
        factored = (True, True, True, False),
    ):
        super().__init__()
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.cumulative = torch.nn.Sequential(*self.get_cumulative_modules(channels, features, factored))

    def forward(self, x):
        upper = self.cumulative(x + 0.5).sigmoid()
        lower = self.cumulative(x - 0.5).sigmoid()
        return NegativeLogarithmFunction.apply(upper - lower, self.epsilon).sum(dim=[1, 2, 3])

    def get_cumulative_modules(self, channels, features, factored):
        return [FactorizedCumulative(channels, features[i], features[i+1], factored[i]) for i in range(len(factored))]

class Autoencoder(torch.nn.Module):
    def __init__(
        self,
        features = 3,
        channels = 192,
        experts = 16,
        capacity = 1,
        minimum = 1e-6,
        origin = 0.1,
        epsilon = 1e-6,
    ):
        super().__init__()
        self.encoder = Encoder(features, channels, experts=experts, capacity=capacity, minimum=minimum, origin=origin, epsilon=epsilon)
        self.hyper_encoder = HyperEncoder(channels)
        self.quantizer = Quantizer()
        self.hyper_decoder = HyperDecoder(channels)
        self.decoder = Decoder(channels, features, experts=experts, capacity=capacity, minimum=minimum, origin=origin, epsilon=epsilon)
        self.gaussian_likelihood = GaussianLikelihood(epsilon=epsilon)
        self.factorized_likelihood = FactorizedLikelihood(channels, epsilon=epsilon)

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_symbols = self.quantizer(x_encoded)
        x_decoded = self.decoder(x_symbols)
        y_encoded = self.hyper_encoder(x_encoded)
        y_symbols = self.quantizer(y_encoded)
        y_decoded = self.hyper_decoder(y_symbols).chunk(2, dim=1)
        return x_decoded, self.gaussian_likelihood(x_symbols, y_decoded[0], y_decoded[1]) + self.factorized_likelihood(y_symbols)

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
    model = Autoencoder(features=MODEL_FEATURES, channels=MODEL_CHANNELS, experts=MODEL_EXPERTS, capacity=MODEL_CAPACITY).to(device)
    bpp_metric = torchmetrics.MeanMetric().to(device)
    psnr_metric = torchmetrics.image.PeakSignalNoiseRatio(1.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), SCRIPT_TRAINING_LEARNING_RATE)
    epoch_offset = 0
    testing_dataset = Dataset(DATASET_TESTING_DIRECTORY, patch_size=DATASET_PATCH_SIZE)
    training_dataset = Dataset(DATASET_TRAINING_DIRECTORY, patch_size=DATASET_PATCH_SIZE)
    testing_batches = torch.utils.data.DataLoader(testing_dataset, batch_size=SCRIPT_TRAINING_BATCH_SIZE, shuffle=False)
    training_batches = torch.utils.data.DataLoader(training_dataset, batch_size=SCRIPT_TRAINING_BATCH_SIZE, shuffle=True)

    if SCRIPT_CHECKPOINT_IMPORT is not None:
        checkpoint = torch.load(SCRIPT_CHECKPOINT_IMPORT, map_location=device, weights_only=True)
        epoch_offset = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    for epoch in range(epoch_offset + 1, SCRIPT_TRAINING_EPOCHS + 1):
        model.train()
        bpp_metric.reset()
        psnr_metric.reset()
        for samples in training_batches:
            samples = samples.to(device).float() * DATASET_PATCH_SCALING
            outputs, bits = model(samples)
            rate_loss = bits.mean() / DATASET_PATCH_PIXELS
            distortion_loss = torch.nn.functional.mse_loss(outputs, samples) * DATASET_PATCH_UNSCALING
            (rate_loss + distortion_loss * MODEL_LAMBDA_MULTIPLIER).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SCRIPT_TRAINING_GRADIENT_CLIPPING_THRESHOLD)
            optimizer.step()
            optimizer.zero_grad()
            bpp_metric.update(bits / DATASET_PATCH_PIXELS)
            psnr_metric.update(outputs.clamp(0.0, 1.0), samples)
        bpp_metric_output = bpp_metric.compute().item()
        psnr_metric_output = psnr_metric.compute().item()
        print(f"[EPOCH {epoch}]: BPP: {bpp_metric_output:.5f} PSNR: {psnr_metric_output:.3f}")
        with open(f"outputs/{MODEL_UNIQUE_VARIANT_STRING}-TRAINING.csv", mode="a", newline="") as file:
            csv.writer(file).writerow([epoch, bpp_metric_output, psnr_metric_output])
        if epoch % SCRIPT_CHECKPOINT_SAVING_INTERVAL == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, f"outputs/{MODEL_UNIQUE_VARIANT_STRING}-{epoch}.pth")
        if epoch % SCRIPT_TESTING_INTERVAL == 0:
            model.eval()
            bpp_metric.reset()
            psnr_metric.reset()
            with torch.no_grad():
                for _ in range(SCRIPT_TESTING_ITERATIONS):
                    for samples in testing_batches:
                        samples = samples.to(device).float() * DATASET_PATCH_SCALING
                        outputs, bits = model(samples)
                        bpp_metric.update(bits / DATASET_PATCH_PIXELS)
                        psnr_metric.update(outputs.clamp(0.0, 1.0), samples)
                with open(f"outputs/{MODEL_UNIQUE_VARIANT_STRING}-TESTING.csv", mode="a", newline="") as file:
                    csv.writer(file).writerow([epoch, bpp_metric.compute().item(), psnr_metric.compute().item(), SCRIPT_TESTING_ITERATIONS])
