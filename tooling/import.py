import os
import sys
import random
from PIL import Image 

WIDTH = 256
HEIGHT = 256
SAMPLES = 10

def save_random_image_samples(image, filename, directory, width, height, samples):
    for _ in range(samples):
        x = random.randint(0, image.width - width)
        y = random.randint(0, image.height - height)
        image.crop((x, y, x + width, y + height)).save(os.path.join(directory, f"{filename}_X{x}Y{y}.png"))

if __name__ == "__main__":
    images, output = sys.argv[1], sys.argv[2]
    for root, _, files in os.walk(images):
        for file in files:
            filename, extension = os.path.splitext(file)
            if not extension.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            with Image.open(os.path.join(root, file)) as image:
                if image.width >= WIDTH and image.height >= HEIGHT:
                    save_random_image_samples(image.convert("RGB"), filename, output, WIDTH, HEIGHT, SAMPLES)
