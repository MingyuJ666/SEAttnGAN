

import os
import sys



current_cwd = os.getcwd()
src_path = '/'.join(current_cwd.split('/')[:-1])
sys.path.append(src_path)




import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.stats import entropy
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm

from sample import prepare_data, generate_images
from src.generator.model import Generator
from src.text_encoder.model import RNNEncoder
from utils import create_loader




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")





class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True).to(self.device)
        print(self.model.fc)
        self.linear = self.model.fc
        self.model.fc, self.model.dropout = [nn.Sequential()] * 2
      
    @torch.no_grad()
    def get_last_layer(self, x):
        x = F.interpolate(x, size=300, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.model(x)





classifier = InceptionV3().to(device)
classifier = classifier.eval()




batch_size = 32
test_loader = create_loader(256, batch_size, "../data", "test")
n_words = test_loader.dataset.n_words




generator = Generator(n_channels=32, latent_dim=100).to(device)
generator.load_state_dict(torch.load("../gen_weights/gen_epoch_310.pth", map_location=device))
generator = generator.eval()




text_encoder = RNNEncoder.load("../text_encoder_weights/text_encoder200.pth", n_words)
text_encoder.to(device)

for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder = text_encoder.eval()



def calculate_fid(repr1, repr2):
    # shape of reprs: (-1, embed_dim)
    
    # shape of mus: (embed_dim, )
    mu_r, mu_g = np.mean(repr1, axis=0), np.mean(repr2, axis=0)
    # rowvar=False:
    #     each column represents a variable, while the rows contain observations
    # shape of sigmas: (embed_dim, embed_dim)
    sigma_r, sigma_g = np.cov(repr1, rowvar=False), np.cov(repr2, rowvar=False)
    
    diff = mu_r - mu_g
    diff_square_norm = diff.dot(diff)
    
    product = sigma_r.dot(sigma_g)
    sqrt_product, _ = sqrtm(product, disp=False)
    

    if not np.isfinite(sqrt_product).all():
        eye_matrix = np.eye(sigma_r.shape[0]) * 1e-8
        sqrt_product = linalg.sqrtm((sigma_r + eye_matrix).dot(sigma_g + eye_matrix))
    
    # np.iscomplexobj:
    #     Check for a complex type or an array of complex numbers.
    #     The return value, True if x is of a complex type
    #     or has at least one complex element.
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    fid = diff_square_norm + np.trace(sigma_r + sigma_g - 2 * sqrt_product)
    
    return fid




def build_representations():
    real_reprs = np.zeros((len(test_loader) * batch_size, 2048))
    fake_reprs = np.zeros((len(test_loader) * batch_size, 2048))
    
    for i, batch in enumerate(tqdm(test_loader, desc="Build representations")):
        images, captions, captions_len, file_names = prepare_data(batch, device)
        sent_emb, word = text_encoder(captions, captions_len)
        sent_emb = sent_emb.detach()
        fake_images = generate_images(generator, sent_emb, device)

        clf_out_real = classifier.get_last_layer(images)
        clf_out_fake = classifier.get_last_layer(fake_images)


        real_reprs[i * batch_size: (i + 1) * batch_size] = clf_out_real.cpu().numpy()
        fake_reprs[i * batch_size: (i + 1) * batch_size] = clf_out_fake.cpu().numpy()
            
    return real_reprs, fake_reprs





real_values, fake_values = build_representations()
real_values = torch.tensor(real_values)
fake_values = torch.tensor(fake_values)



fid_value = calculate_fid(real_values, fake_values)
print(f"FID value = {fid_value}")






def inception_score(reprs, batch_size):
    def get_pred(x):
        x = classifier.linear(torch.tensor(x, dtype=torch.float))
        return F.softmax(x).data.cpu().numpy()


    preds = np.zeros((reprs.shape[0], 1000))

    splits = 0
    for i in range(0, len(preds), batch_size):
        aaai = reprs[i:i + batch_size]
        aai = torch.tensor(aaai)
        aai = aai.to(device)
        z = get_pred(aai)
        preds[i:i + batch_size] = z
        splits += 1
    
    split_scores = []

    for k in range(splits):
        part = preds[k * batch_size: (k+1) * batch_size, :]
        py = np.mean(part, axis=0)
        
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
            
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)





a,b = inception_score(fake_values, batch_size)
print('is',a)
print('is',b)

