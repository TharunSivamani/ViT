import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Turns a 2D input image into a 1D sequence learnable embedding vector.
    """
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super(PatchEmbedding, self).__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.patcher = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.embedding_dim,
            kernel_size = self.patch_size,
            stride = patch_size
        )

        self.flatten = nn.Flatten(
            start_dim = 2,
            end_dim = 3
        )

    def forward(self, x):

        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Image Size must be divisible by patch size, given image shape: {image_resolution}, patch_size: {self.patch_size}"

        x = self.patcher(x)
        x = self.flatten(x)

        return x.permute(0, 2, 1) # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]
