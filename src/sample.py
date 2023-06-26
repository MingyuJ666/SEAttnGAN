import os

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from src.generator.model import Generator
from src.objects.utils import prepare_data
from src.text_encoder.model import RNNEncoder


@torch.no_grad()
def generate_images(generator: Generator, sentence_embeds: Tensor,
                    device: torch.device) -> Tensor:
    batch_size = sentence_embeds.shape[0]
    noise = torch.randn(batch_size, 100).to(device)
    return generator(noise, sentence_embeds)


def save_image(image: np.ndarray, save_dir: str, file_name: str):
    # [-1, 1] --> [0, 255]
    image = (image + 1.0) * 127.5
    image = image.astype(np.uint8)
    image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image)
    fullpath = os.path.join(save_dir, f"{file_name.replace('/', '_')}.png")
    image.save(fullpath)


def sample(generator: Generator, text_encoder: RNNEncoder, batch, save_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(save_dir, exist_ok=True)

    images, captions, captions_len, file_names = prepare_data(batch, device)
    sent_emb = text_encoder(captions, captions_len).detach()

    fake_images = generate_images(generator, sent_emb, device)

    for i in range(images.shape[0]):
        im = fake_images[i].data.cpu().numpy()
        save_image(im, save_dir, file_names[i])
