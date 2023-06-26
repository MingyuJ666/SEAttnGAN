import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)


def func_attention(query, context, gamma1):
    """
    query: batch x ndf x queryL
    context: batch x ndf x ih x iw (sourceL=ihxiw)
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)  # Eq. (7) in AttnGAN paper
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax()(attn)  # Eq. (8)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    #  Eq. (9)
    attn = attn * gamma1
    attn = nn.Softmax()(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x queryL x idf
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)

        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> batch*queryL x sourceL
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        # (batch x idf x sourceL)(batch x sourceL x queryL)
        # --> batch x idf x queryL
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)

        return weightedContext, attn


# from src.generator.residual_block import ResidualBlockG
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
        print('sss',sentence_embed.shape)
        scale_param = self.gamma_mlp(sentence_embed)
        shift_param = self.beta_mlp(sentence_embed)
        print('www',scale_param.shape)

        scale_param = scale_param.unsqueeze(-1).unsqueeze(-1).expand(x.shape)
        shift_param = shift_param.unsqueeze(-1).unsqueeze(-1).expand(x.shape)

        return scale_param * x + shift_param


class ResidualBlockG(nn.Module):
    def __init__(self, df_1_c_out: int, df_2_c_out: int, affine_h_d: int = 256,
                 df_1_c_in: int = 256, df_2_c_in: int = 256):
        super().__init__()
        # DFBlock_1
        self.fusion_block_1 = AffineBlock(input_dim=df_1_c_in, hidden_dim=affine_h_d, output_dim=df_1_c_out)
        self.fusion_block_2 = AffineBlock(input_dim=df_1_c_in, hidden_dim=affine_h_d, output_dim=df_1_c_out)
        self.conv_1 = nn.Conv2d(df_1_c_out, df_2_c_out, kernel_size=3, stride=1, padding=1)

        # DFBlock_2
        self.fusion_block_3 = AffineBlock(input_dim=df_2_c_in, hidden_dim=affine_h_d, output_dim=df_2_c_out)
        self.fusion_block_4 = AffineBlock(input_dim=df_2_c_in, hidden_dim=affine_h_d, output_dim=df_2_c_out)
        self.conv_2 = nn.Conv2d(df_2_c_out, df_2_c_out, kernel_size=3, stride=1, padding=1)

        self.scale_conv = None
        if df_1_c_out != df_2_c_out:
            self.scale_conv = nn.Conv2d(df_1_c_out, df_2_c_out, kernel_size=1, stride=1, padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.scale_conv is not None:
            x = self.scale_conv(x)

        return x

    def _df_block_1(self, x: Tensor, sentence_embed: Tensor) -> Tensor:
        h = self.fusion_block_1(x, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.fusion_block_2(h, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        return self.conv_1(h)

    def _df_block_2(self, x: Tensor, sentence_embed: Tensor) -> Tensor:
        h = self.fusion_block_3(x, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        h = self.fusion_block_4(h, sentence_embed)
        h = F.leaky_relu(h, 0.2, inplace=True)
        return self.conv_2(h)

    def _residual(self, x: Tensor, y):
        # DFBlock_1
        h_1 = self._df_block_1(x, y)

        # DFBlock_2
        h_2 = self._df_block_2(h_1, y)

        return h_2

    def forward(self, x, sentence_embed: Tensor) -> Tensor:
        return self._shortcut(x) + self.gamma * self._residual(x, sentence_embed)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU())
    return block


class Generator(nn.Module):
    def __init__(self, n_channels: int, latent_dim: int):
        super().__init__()

        self.n_channels = n_channels

        self.linear_in = nn.Linear(latent_dim, n_channels * 8 * 4 * 4)

        self.res_blocks = nn.ModuleList([
            ResidualBlockG(8 * n_channels, 8 * n_channels),
            ResidualBlockG(8 * n_channels, 4 * n_channels),
            ResidualBlockG(4 * n_channels, 2 * n_channels),
            ResidualBlockG(2 * n_channels, 1 * n_channels),
            ResidualBlockG(1 * n_channels, 1 * n_channels),

        ])

        self.res_block_out = ResidualBlockG(2 * n_channels, 1 * n_channels)

        self.att = GlobalAttentionGeneral(32, 256)
        self.upsample = upBlock(64, 64)

        self.conv_out = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_channels, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )



    def forward(self, noise_vector: Tensor, sentence_embed: Tensor, word_embed: Tensor) -> Tensor:
       # noise_vector.shape = [bs, latent_dim]
       out = self.linear_in(noise_vector)
       # out.shape = [bs, n_channels * 8, 4, 4]
       out = out.view(noise_vector.size(0), 8 * self.n_channels, 4, 4)
       for res_block in self.res_blocks:
          out = F.interpolate(res_block(out, sentence_embed), scale_factor=2)

       out_attn1, att = self.att(out, word_embed)
       out = torch.cat((out, out_attn1), 1)
       out = self.upsample(out)
       out = self.res_block_out(out, sentence_embed)

       image = self.conv_out(out)

       return image


