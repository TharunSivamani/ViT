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
        print(f"After Con2d layer x shape: {x.shape}") # torch.Size([1, 768, 14, 14])
        x = self.flatten(x)
        print(f"After Flattening x shape: {x.shape}") # torch.Size([1, 768, 196])
        return x.permute(0, 2, 1) # [batch_size, P^2•C, N] -> [batch_size, N, P^2•C]


class MultiheadSelfAttentionBlock(nn.Module):
    """
    Creates a multi-head self-attention block 
    """
    def __init__(self, embedding_dim:int=768, num_heads:int=12, attn_dropout:float=0):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) 
        
    def forward(self, x):
        x = self.layer_norm(x)
        print(f"After Layer-Norm x shape: {x.shape}")
        attn_output, _ = self.multihead_attn(query=x, 
                                             key=x, 
                                             value=x,
                                             need_weights=False)
        print(f"Self-Attn shape: {attn_output.shape}")
        return attn_output


if __name__ == "__main__":

    patch = PatchEmbedding(3, 16, 768)
    x = torch.randn((1, 3, 224, 224))

    out = patch(x)
    print(out.shape)