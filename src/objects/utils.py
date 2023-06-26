from typing import List, Tuple

import torch
from torch import Tensor


def prepare_data(batch: Tuple[Tensor, Tensor, Tensor, Tuple[str]],
                 device: torch.device) -> Tuple[Tensor, Tensor, Tensor, List[str]]:
    images, captions, captions_len, file_names = batch

    sorted_cap_lens, sorted_cap_indices = torch.sort(captions_len, 0, True)
    sorted_cap_lens = sorted_cap_lens.to(device)

    sorted_images = images[sorted_cap_indices].to(device)
    sorted_captions = captions[sorted_cap_indices].squeeze().to(device)
    sorted_file_names = [file_names[i] for i in sorted_cap_indices.numpy()]

    return sorted_images, sorted_captions, sorted_cap_lens, sorted_file_names
