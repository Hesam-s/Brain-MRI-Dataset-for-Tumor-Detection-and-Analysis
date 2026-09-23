import torch
import torch.nn as nn

class HybridViTGRU(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=2):
        super().__init__()

        embed_dim = 64
        num_heads = 4
        gru_hidden = 1024

        self.num_patches = (image_size // patch_size) ** 2
        
        # 1. Patch Embedding via Convolution
        self.patch_embed = nn.Conv2d(
            3, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # 2. Position Embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

        # 3. Transformer Encoder Blocks (8 Layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=8
        )

        # 4. GRU Layer for Sequential Context
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            batch_first=True
        )

        # 5. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x shape: (B, 3, 224, 224)
        x = self.patch_embed(x)              # (B, 64, 14, 14)
        
        # Flatten H and W into sequence length (196) and move embed_dim to end
        x = x.flatten(2).transpose(1, 2)     # (B, 196, 64)

        # Add Position Embeddings (Broadcasting handles Batch dim)
        x = x + self.pos_embed

        # Transformer Processing
        x = self.transformer(x)              # (B, 196, 64)
        
        # GRU Processing
        # hidden shape: (num_layers=1, B, 1024)
        _, hidden = self.gru(x)
        
        # Use the last hidden state for classification
        return self.classifier(hidden[-1])