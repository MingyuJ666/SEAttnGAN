
import os
import sys





current_cwd = os.getcwd()
src_path = '/'.join(current_cwd.split('/')[:-1])
sys.path.append(src_path)





import numpy as np
import torch

from sample import sample, save_image
from src.generator.model import Generator
from src.text_encoder.model import RNNEncoder
from utils import create_loader





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




import time
t = time.time()

generator = Generator(n_channels=32, latent_dim=100).to(device)
generator.load_state_dict(torch.load("../gen_weights/gen_360.pth", map_location=device))
generator = generator.eval()





train_loader = create_loader(256, 24, "../data", "test")




n_words = train_loader.dataset.n_words





text_encoder = RNNEncoder.load("../text_encoder_weights/text_encoder200.pth", n_words)
text_encoder.to(device)

for p in text_encoder.parameters():
    p.requires_grad = False
text_encoder = text_encoder.eval()





dataset = train_loader.dataset




def gen_own_bird(word_caption, name,i):
    codes = [dataset.word2code[w] for w in word_caption.lower().split()]
    
    caption = np.array(codes)
    pad_caption = np.zeros((18, 1), dtype='int64')

    if len(caption) <= 18:
        pad_caption[:len(caption), 0] = caption
        len_ = len(caption)
    else:
        indices = list(np.arange(len(caption)))
        np.random.shuffle(indices)
        pad_caption[:, 0] = caption[np.sort(indices[:18])]
        len_ = 18
    tensor1 = torch.tensor(pad_caption).reshape(1, -1).to(device)
    tensor2 = torch.tensor([len_]).to(device)
    embed,word = text_encoder(tensor1, tensor2)

    batch_size = embed.shape[0]
    noise = torch.randn(batch_size, 100).to(device)
    img =  generator(noise, embed, word)
    save_image(img[0].data.cpu().numpy(), "../gen_images", name + str(i))




caption = "A blue bird with black eyes"
i=1
gen_own_bird(caption, caption,i)
print(time.time()-t)
