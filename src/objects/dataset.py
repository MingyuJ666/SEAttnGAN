import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.random as random
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class DFGANDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", transform: Optional[Compose] = None):
        self.split = split
        self.data_dir = data_dir

        self.split_dir = os.path.join(data_dir, split)
        self.captions_path = os.path.join(self.data_dir, "captions.pickle")
        self.filenames_path = os.path.join(self.split_dir, "filenames.pickle")

        self.transform = transform

        self.embeddings_num = 10

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.images_dir = os.path.join(self.data_dir, "CUB_200_2011/CUB_200_2011/images")
        self.bbox_path = os.path.join(self.data_dir, "CUB_200_2011/CUB_200_2011/bounding_boxes.txt")
        self.images_path = os.path.join(self.data_dir, "CUB_200_2011/CUB_200_2011/images.txt")

        self.bbox = self._load_bbox()

        self.file_names, self.captions, self.code2word, self.word2code = self._load_text_data()

        self.n_words = len(self.code2word)
        self.num_examples = len(self.file_names)

        self._print_info()

    def _print_info(self):
        print(f"Total filenames: {len(self.bbox)}")
        print(f"Load captions from: {self.captions_path}")
        print(f"Load file names from: {self.filenames_path} ({self.num_examples})")
        print(f"Dictionary size: {self.n_words}")
        print(f"Embeddings number: {self.embeddings_num}")

    def _load_bbox(self) -> Dict[str, List[int]]:
        df_bbox = pd.read_csv(self.bbox_path, delim_whitespace=True, header=None).astype(int)

        df_image_names = pd.read_csv(self.images_path, delim_whitespace=True, header=None)
        image_names = df_image_names[1].tolist()

        filename_bbox = dict()
        for i, file_name in enumerate(image_names):
            bbox = df_bbox.iloc[i][1:].tolist()
            filename_bbox[file_name[:-4]] = bbox

        return filename_bbox

    def _load_text_data(self) -> Tuple[List[str], List[List[int]],
                                       Dict[int, str], Dict[str, int]]:
        with open(self.captions_path, 'rb') as file:
            train_captions, test_captions, code2word, word2code = pickle.load(file)

        filenames = self._load_filenames()

        if self.split == 'train':
            return filenames, train_captions, code2word, word2code

        return filenames, test_captions, code2word, word2code

    def _load_filenames(self) -> List[str]:
        if os.path.isfile(self.filenames_path):
            with open(self.filenames_path, 'rb') as file:
                return pickle.load(file)

        raise ValueError(f"File {self.filenames_path} does not exist")

    def _get_caption(self, caption_idx: int) -> Tuple[np.ndarray, int]:
        caption = np.array(self.captions[caption_idx])
        pad_caption = np.zeros((18, 1), dtype='int64')

        if len(caption) <= 18:
            pad_caption[:len(caption), 0] = caption
            return pad_caption, len(caption)

        indices = list(np.arange(len(caption)))
        np.random.shuffle(indices)
        pad_caption[:, 0] = caption[np.sort(indices[:18])]

        return pad_caption, 18

    def _get_image(self, image_path: str, bbox: List[int]) -> Tensor:

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)

        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)

        image = image.crop((x1, y1, x2, y2))
        image = self.normalize(self.transform(image))

        return image

    def _get_random_caption(self, idx: int) -> Tuple[np.ndarray, int]:
        caption_shift = random.randint(0, self.embeddings_num)
        caption_idx = idx * self.embeddings_num + caption_shift
        return self._get_caption(caption_idx)

    def __getitem__(self, idx: int) -> Tuple[Tensor, np.ndarray, int, str]:
        file_name = self.file_names[idx]
        image = self._get_image(f"{self.images_dir}/{file_name}.jpg", self.bbox[file_name])

        encoded_caption, caption_len = self._get_random_caption(idx)

        return image, encoded_caption, caption_len, file_name

    def __len__(self) -> int:
        return self.num_examples
