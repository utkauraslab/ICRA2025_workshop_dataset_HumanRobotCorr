pip install perceiver-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# self_define Perceiver model
class SimplePerceiverEncoder(nn.Module):
    def __init__(self, input_dim=64, latent_dim=512, num_latents=2048, num_self_attn_layers=6):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))  # (2048, 512)

        # cross-attention: Latents attend to voxel input
        self.cross_attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=8, batch_first=True)
        self.cross_ln = nn.LayerNorm(latent_dim)

        # self-attention layers
        self.self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=latent_dim, nhead=8, batch_first=True)
            for _ in range(num_self_attn_layers)
        ])

        # voxel input embedding (project 64→512)
        self.input_proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        # x: (B, N, 64)
        B, N, _ = x.shape
        x = self.input_proj(x)  # → (B, N, 512)

        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)  # (B, 2048, 512)

        # cross-attn: latents attend to x
        attn_out, _ = self.cross_attn(latents, x, x)
        latents = self.cross_ln(latents + attn_out)

        for layer in self.self_attn_layers:
            latents = layer(latents)

        return latents  # (B, 2048, 512)


# classifier model
class PerceiverClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SimplePerceiverEncoder(input_dim=64, latent_dim=512, num_latents=2048)
        self.classifier = nn.Linear(512, 8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        latents = self.encoder(x)         # (B, 2048, 512)
        pooled = latents.mean(dim=1)      # (B, 512)
        logits = self.classifier(pooled)  # (B, 8)
        return self.softmax(logits)       # (B, 8)

# output voxel_grid
voxel_grid = torch.randn(1, 100, 100, 100, 10)  

# reshape 
B, D, H, W, C = voxel_grid.shape
voxel_input = voxel_grid.reshape(B, -1, C)  # <-(B, N, C)

# give perceiver model
output = model(voxel_input)  # -> (B, 8)

print("Output shape:", output.shape)
print("Output probabilities:", output)
