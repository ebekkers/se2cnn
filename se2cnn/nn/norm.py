from torch import nn

class SE2LayerNorm(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, Theta, H, W) -> (N, Theta, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # (N, Theta, H, W, C) -> (N, C, Theta, H, W)
        return x