
class Generator(nn.Module):
    def __init__(self, n_channels: int, latent_dim: int):
        super().__init__()

        self.n_channels = n_channels

        self.linear_in = nn.Linear(latent_dim, n_channels * 8 * 4 * 4)

        self.res_blocks = nn.ModuleList([
            # ResidualBlockG(8 * n_channels, 8 * n_channels),
            # ResidualBlockG(8 * n_channels, 8 * n_channels),
            # ResidualBlockG(8 * n_channels, 8 * n_channels),
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

        self.res_block = nn.ModuleList([
            # ResidualBlockG(8 * n_channels, 8 * n_channels),
            # ResidualBlockG(8 * n_channels, 8 * n_channels),
            # ResidualBlockG(8 * n_channels, 8 * n_channels),
            ResidualBlockG(8 * n_channels, 8 * n_channels),
            ResidualBlockG(8 * n_channels, 4 * n_channels),
            ResidualBlockG(4 * n_channels, 2 * n_channels),
            ResidualBlockG(2 * n_channels, 1 * n_channels),

        ])




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


def forward(self, noise_vector: Tensor, sentence_embed: Tensor, word_embed: Tensor) -> Tensor:
        out = self.linear_in(noise_vector)

        out = out.view(noise_vector.size(0), 8 * self.n_channels, 4, 4)
        for res_block in self.res_block:
            out = F.interpolate(res_block(out, sentence_embed), scale_factor=2)

        out_attn1, att = self.att(out, word_embed)
        out = torch.cat((out, out_attn1), 1)

        out = self.upsample(out)
        out = self.upsample(out)
        out = self.res_block_out(out, sentence_embed)

        image = self.conv_out(out)

        return image


