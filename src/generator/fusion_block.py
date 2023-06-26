import torch.nn as nn
from torch import Tensor


class AffineBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gamma_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self.beta_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

        self._xavier_normal_initialization()

    def _xavier_normal_initialization(self):
        for module in self.gamma_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, val=0)

        for module in self.beta_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, val=0)

    def forward(self, x: Tensor, sentence_embed: Tensor) -> Tensor:
        scale_param = self.gamma_mlp(sentence_embed)
        shift_param = self.beta_mlp(sentence_embed)

        scale_param = scale_param.unsqueeze(-1).unsqueeze(-1).expand(x.shape)
        shift_param = shift_param.unsqueeze(-1).unsqueeze(-1).expand(x.shape)

        return scale_param * x + shift_param
