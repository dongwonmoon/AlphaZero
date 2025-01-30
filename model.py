import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.dropout(x)
        if residual.shape != x.shape:
            residual = F.interpolate(residual, size=x.shape[2:], mode="nearest")
        return F.relu(x + residual)


class AlphaZeroNet(nn.Module):
    def __init__(
        self, board_size, action_size, num_res_blocks, in_channels, mid_channels
    ):
        super(AlphaZeroNet, self).__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.conv1 = nn.Conv2d(14, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(in_channels, mid_channels, in_channels)
                for _ in range(num_res_blocks)
            ]
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 32),
            nn.Mish(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


if __name__ == "__main__":
    model = AlphaZeroNet(8, 4032, 5, 64, 16)
    summary(model, (8, 8, 14))
