import torch
from dataclasses import dataclass

@dataclass
class Config:
    image_size: int = 224
    d_model: int = 256
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    n_heads: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_seq_len: int = 768
    fourier_L: int = 10
    max_step: float = 0.09
    # kolam19 coordinate bounds: [-18.5, 18.5]
    coord_min: float = -18.5
    coord_max: float = 18.5
    coord_loss_type: str = "huber"
    huber_beta: float = 0.1
    coord_weight: float = 100.0
    eos_weight: float = 0.1
    curvature_weight: float = 0.01
    closed_loop_weight: float = 0.05
    diagnostic_mode: bool = True
    teacher_forcing_start: float = 1.0
    teacher_forcing_end: float = 0.0
    teacher_forcing_decay_steps: int = 2000
    warmup_steps: int = 1000
    base_lr: float = 1e-4
    grad_clip_norm: float = 1.0
    batch_size: int = 16
    num_epochs: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
