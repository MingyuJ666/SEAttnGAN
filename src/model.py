import os.path
from typing import Tuple, List
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import trange

from discriminator.model import Discriminator
from generator.model import Generator
from objects.utils import prepare_data
from text_encoder.model import RNNEncoder

class DeepFusionGAN:
    def __init__(self, n_words, encoder_weights_path: str, image_save_path: str, gen_path_save: str):
        super().__init__()
        self.image_save_path = image_save_path
        self.gen_path_save = gen_path_save

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator = Generator(n_channels=32, latent_dim=100).to(self.device)
        self.discriminator = Discriminator(n_c=32).to(self.device)

        self.text_encoder = RNNEncoder.load(encoder_weights_path, n_words)
        self.text_encoder.to(self.device)

        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_encoder.eval()

        # self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.0, 0.9))
        # self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0004, betas=(0.0, 0.9))
        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=0.0002,betas=(0.5, 0.99))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=0.0008, betas=(0.5, 0.99))

        # self.g_optim =  torch.optim.SGD(self.generator.parameters(), lr = 0.0002, momentum=0.9)
        # self.d_optim =  torch.optim.SGD(self.discriminator.parameters(), lr = 0.0002, momentum=0.9)
        self.relu = nn.ReLU()

    def _zero_grad(self):
        self.d_optim.zero_grad()
        self.g_optim.zero_grad()

    def _compute_gp(self, images: Tensor, sentence_embeds: Tensor) -> Tensor:
        batch_size = images.shape[0]

        images_interpolated = images.data.requires_grad_()
        sentences_interpolated = sentence_embeds.data.requires_grad_()

        embeds = self.discriminator.build_embeds(images_interpolated)
        logits = self.discriminator.get_logits(embeds, sentences_interpolated)

        grad_outputs = torch.ones_like(logits)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=(images_interpolated, sentences_interpolated),
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=True
        )

        grad_0 = grads[0].reshape(batch_size, -1)
        grad_1 = grads[1].reshape(batch_size, -1)

        grad = torch.cat((grad_0, grad_1), dim=1)
        grad_norm = grad.norm(2, 1)

        return grad_norm

    def fit(self, train_loader: DataLoader, num_epochs: int = 500) -> Tuple[List[float], List[float], List[float]]:
        g_losses_epoch, d_losses_epoch, d_gp_losses_epoch = [], [], []
        for epoch in trange(num_epochs, desc="Train Deep Fusion GAN"):

            g_losses, d_losses, d_gp_losses = [], [], []
            for batch in train_loader:
                images, captions, captions_len, _ = prepare_data(batch, self.device)
                batch_size = images.shape[0]


                sentence_embeds, words_embs = self.text_encoder(captions, captions_len)
                sentence_embeds, words_embs = sentence_embeds.detach(), words_embs.detach()
                #sentence_embeds, word_embeds = self.text_encoder(captions, captions_len).detach()

                real_embeds = self.discriminator.build_embeds(images)
                real_logits = self.discriminator.get_logits(real_embeds, sentence_embeds)
                d_loss_real = self.relu(1.0 - real_logits).mean()

                shift_embeds = real_embeds[:(batch_size - 1)]
                shift_sentence_embeds = sentence_embeds[1:batch_size]
                shift_real_image_embeds = self.discriminator.get_logits(shift_embeds, shift_sentence_embeds)

                d_loss_mismatch = self.relu(1.0 + shift_real_image_embeds).mean()

                noise = torch.randn(batch_size, 100).to(self.device)


                fake_images = self.generator(noise, sentence_embeds, words_embs)

                fake_embeds = self.discriminator.build_embeds(fake_images.detach())
                fake_logits = self.discriminator.get_logits(fake_embeds, sentence_embeds)

                d_loss_fake = self.relu(1.0 + fake_logits).mean()

                d_loss = d_loss_real + (d_loss_fake + d_loss_mismatch) / 2.0

                self._zero_grad()
                d_loss.backward()
                self.d_optim.step()

                d_losses.append(d_loss.item())

                grad_l2norm = self._compute_gp(images, sentence_embeds)
                d_loss_gp = 2.0 * torch.mean(grad_l2norm ** 6)

                self._zero_grad()
                d_loss_gp.backward()
                self.d_optim.step()

                d_gp_losses.append(d_loss_gp.item())

                fake_embeds = self.discriminator.build_embeds(fake_images)
                fake_logits = self.discriminator.get_logits(fake_embeds, sentence_embeds)
                g_loss = -fake_logits.mean()

                self._zero_grad()
                g_loss.backward()
                self.g_optim.step()

                g_losses.append(g_loss.item())


            g_losses_epoch.append(np.mean(g_losses))
            d_losses_epoch.append(np.mean(d_losses))
            d_gp_losses_epoch.append(np.mean(d_gp_losses))


            if epoch % 1 == 0:
             self._save_fake_image(fake_images, epoch)
             self._save_gen_weights(epoch)


            print('g_losses_epoch', g_losses_epoch)
            print('d_losses_epoch', d_losses_epoch)
            print('gp_losses_epoch', d_gp_losses_epoch)



        return g_losses_epoch, d_losses_epoch, d_gp_losses_epoch

    def _save_fake_image(self, fake_images: Tensor, epoch: int):
        img_path = os.path.join(self.image_save_path, f"fake_samplenormal_epoch_{epoch}.png")
        vutils.save_image(fake_images.data, img_path, normalize=True)

    def _save_gen_weights(self, epoch: int):
        gen_path = os.path.join(self.gen_path_save, f"gennormal_{epoch}.pth")
        torch.save(self.generator.state_dict(), gen_path)
