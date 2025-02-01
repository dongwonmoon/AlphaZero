import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def init_weights(m: nn.Module) -> None:
    """
    Applies weight initialization to the layers of the neural network.
    Uses Kaiming normalization for Conv2d and Xavier for Linear by default.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ResidualBlock(nn.Module):
    """
    A custom Residual Block that prepares an output with the same shape
    as the input. Extra Conv + BN + Dropout are used for regularization.
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.layer(x)
        # If the shape changed, resize residual
        if residual.shape != out.shape:
            residual = F.interpolate(residual, size=out.shape[2:], mode="nearest")
        return F.relu(out + residual, inplace=True)

class AlphaZeroNet(nn.Module):
    """
    AlphaZero-like network for a chess environment with additional suggested improvements:
      - Weight initialization for stable training.
      - Flexible architecture with residual blocks, policy, and value heads.
    """
    def __init__(
        self,
        board_size: int,
        num_ware: int,      # Equivalent to input_channels (e.g., color planes)
        action_size: int,
        num_res_blocks: int,
        in_channels: int,
        mid_channels: int
    ):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Initial layer maps input (num_ware) to 'in_channels'
        self.layer = nn.Sequential(
            nn.Conv2d(num_ware, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Residual blocks for the main representation
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(in_channels, mid_channels, in_channels)
                for _ in range(num_res_blocks)
            ]
        )

        # Policy head: output a probability distribution over all actions
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 32),
            nn.Mish(),
            nn.Linear(32, action_size),
            nn.Softmax(dim=1),
        )

        # Value head: output a scalar in [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(board_size * board_size, 256),
            nn.Mish(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        # Apply custom weight initialization
        self.apply(init_weights)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input observations of shape (B, H, W, C).
              For chess, typically (B, board_size, board_size, num_planes).
        Returns:
            policy (torch.Tensor): Probability distribution over all possible actions (B, action_size).
            value (torch.Tensor): Scalar value in range [-1, 1] for each batch (B, 1).
        """
        # Move input to device and transform shape to (B, C, H, W)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device, dtype=torch.float32).permute(0, 3, 1, 2)

        # Main representation
        x = self.layer(x)
        x = self.res_blocks(x)

        # Branch into policy & value heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def _test():
    """
    Simple test routine to ensure the network works as expected.
    """
    board_size = 8
    num_ware = 14
    action_size = 4032
    num_res_blocks = 5
    in_channels = 64
    mid_channels = 16

    print("Testing AlphaZeroNet...")
    model = AlphaZeroNet(
        board_size=board_size,
        num_ware=num_ware,
        action_size=action_size,
        num_res_blocks=num_res_blocks,
        in_channels=in_channels,
        mid_channels=mid_channels
    )
    dummy_input = torch.randn(1, board_size, board_size, num_ware)
    policy, value = model(dummy_input)

    print("Policy shape:", policy.shape)   # Expected: (2, 4032)
    print("Value shape:", value.shape)     # Expected: (2, 1)
    print("Forward pass complete.")

    # Summarize model
    summary(model, (board_size, board_size, num_ware))

if __name__ == "__main__":
    _test()