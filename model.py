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


class MLPBlock(nn.Module):
    """
    Creates a Layer Normalized MLP Block
    """
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):

        x = self.layer_norm(x)
        x = self.mlp(x)
        return x
    
class TransformerEncoderBlock(nn.Module):
    """
    Creates a Transformer Encoder Block
    """
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):

        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x

        return x

class ViT(nn.Module):
    """
    Creates a ViT Model
    """
    def __init__(self, img_size=224, in_channels=3, patch_size=16, num_transformer_layer=12, embedding_dim=768, mlp_size=3072, num_heads=12, attn_dropout=0, ml_dropout=0.1, embedding_dropout=0.1, num_classes=1000):
        super().__init__()

        assert img_size % patch_size == 0, f"Image Size must be divisible by patch size, given image shape: {img_size}, patch_size: {patch_size}"
        # (height * width/patch^2)
        self.num_patches = (img_size * img_size) // patch_size ** 2
        self.class_embedding = nn.Parameter(data = torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(data = torch.randn(1, self.num_patches+1, embedding_dim), requires_grad = True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_size=mlp_size, mlp_dropout=ml_dropout) for _ in range(num_transformer_layer)]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
    
    def forward(self, x):

        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat([class_token, x], dim=1)
        x = self.position_embedding(x) + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x[:, 0])

        return x

if __name__ == "__main__":

    patch = PatchEmbedding(3, 16, 768)
    x = torch.randn((1, 3, 224, 224))

    out = patch(x)
    print(out.shape)
