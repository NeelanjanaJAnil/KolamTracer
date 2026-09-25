import torch
import torch.nn as nn
import math

class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 4-stage hierarchical Conv2d network
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, config.d_model, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # 224 -> 112 -> 56 -> 28 -> 14. So P = 14*14 = 196
        self.spatial_pos_embed = nn.Parameter(torch.randn(1, 14 * 14, config.d_model))
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert C == 1 and H == self.config.image_size and W == self.config.image_size, f"Expected (B, 1, {self.config.image_size}, {self.config.image_size}), got {x.shape}"
        
        features = self.conv(x) # (B, d_model, 14, 14)
        B, D, fH, fW = features.shape
        assert D == self.config.d_model, f"Expected d_model {self.config.d_model}, got {D}"
        
        patches = features.flatten(2).transpose(1, 2) # (B, 14*14, d_model)
        P = patches.shape[1]
        
        patches = patches + self.spatial_pos_embed[:, :P, :]
        
        assert patches.shape == (B, P, self.config.d_model), f"Expected (B, {P}, {self.config.d_model}), got {patches.shape}"
        return patches

class FourierCoordEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.L = config.fourier_L
        self.d_model = config.d_model
        # 2 coords (x, y) * L bands * 2 (sin/cos) = 4 * L
        self.proj = nn.Linear(4 * self.L, config.d_model)
        
    def forward(self, coords):
        B, T, D = coords.shape
        assert D == 2, f"Expected input paths (B, T, 2), got {coords.shape}"
        
        bands = 2 ** torch.arange(self.L, dtype=torch.float32, device=coords.device) * math.pi
        bands = bands.view(1, 1, 1, self.L) # (1, 1, 1, L)
        
        coords_expanded = coords.unsqueeze(-1) # (B, T, 2, 1)
        args = coords_expanded * bands # (B, T, 2, L)
        
        sin_emb = torch.sin(args)
        cos_emb = torch.cos(args)
        
        fourier_emb = torch.cat([sin_emb, cos_emb], dim=-1) # (B, T, 2, 2L)
        fourier_emb = fourier_emb.view(B, T, -1) # (B, T, 4L)
        
        out = self.proj(fourier_emb) # (B, T, d_model)
        assert out.shape == (B, T, self.d_model)
        return out

class CausalPathDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.coord_emb = FourierCoordEmbedding(config)
        self.pen_emb = nn.Embedding(2, config.d_model) # 0 = drawing, 1 = EOS/Pad
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        
        self.coord_head = nn.Sequential(
            nn.Linear(config.d_model, 2),
            nn.Tanh()
        )
        self.eos_head = nn.Linear(config.d_model, 1)
        
        # Explicit custom initialization for coord_head
        nn.init.normal_(self.coord_head[0].weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.coord_head[0].bias, 0.0)
        
    def forward(self, memory, target_coords, pen_state, pad_mask=None):
        B, P, D = memory.shape
        assert D == self.config.d_model, f"Expected visual tokens (B, P, d_model), got {memory.shape}"
        
        B_t, T, D_c = target_coords.shape
        assert B == B_t, "Batch size mismatch"
        assert D_c == 2, f"Expected input paths (B, T, 2), got {target_coords.shape}"
        assert pen_state.shape == (B, T), f"Expected pen_state (B, T), got {pen_state.shape}"
        
        positions = torch.arange(T, device=target_coords.device).unsqueeze(0).expand(B, T)
        emb = self.coord_emb(target_coords) + self.pen_emb(pen_state.long()) + self.pos_emb(positions)
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=target_coords.device)
        assert causal_mask.shape == (T, T), f"Expected causal mask (T, T), got {causal_mask.shape}"
        
        if pad_mask is not None:
            assert pad_mask.shape == (B, T), f"Expected pad_mask (B, T), got {pad_mask.shape}"
        
        out = self.transformer_decoder(
            tgt=emb,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask
        )
        
        delta_coords = self.coord_head(out) * self.config.max_step
        eos_logits = self.eos_head(out)
        
        return delta_coords, eos_logits
        
    @torch.no_grad()
    def autoregressive_generate(self, memory, max_len):
        B = memory.shape[0]
        device = memory.device
        
        curr_coords = torch.zeros((B, 1, 2), device=device)
        curr_pen = torch.zeros((B, 1), dtype=torch.long, device=device)
        
        for t in range(max_len - 1):
            delta_coords, eos_logits = self.forward(memory, curr_coords, curr_pen)
            
            next_delta = delta_coords[:, -1:, :]
            next_eos_logit = eos_logits[:, -1:, :]
            
            next_coord = curr_coords[:, -1:, :] + next_delta
            next_pen = (next_eos_logit > 0).long().squeeze(-1)
            
            curr_coords = torch.cat([curr_coords, next_coord], dim=1)
            curr_pen = torch.cat([curr_pen, next_pen], dim=1)
            
        return curr_coords, curr_pen

class KolamModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = VisionEncoder(config)
        self.decoder = CausalPathDecoder(config)
        
    def forward(self, image, target_coords, pen_state, pad_mask=None):
        memory = self.encoder(image)
        delta_coords, eos_logits = self.decoder(memory, target_coords, pen_state, pad_mask)
        return delta_coords, eos_logits
