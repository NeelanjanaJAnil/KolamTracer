import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd


class KolamPathDataset(Dataset):
    def __init__(self, image_paths, csv_paths, config):
        self.image_paths = image_paths
        self.csv_paths = csv_paths
        self.config = config

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('L')  # Grayscale
        orig_w, orig_h = img.size
        img = img.resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0  # (1, H, W)

        # Load coords
        csv_path = self.csv_paths[idx]
        df = pd.read_csv(csv_path)

        # Enforce standard 'x', 'y' headers
        if not {'x', 'y'}.issubset(set(df.columns)):
            if len(df.columns) >= 2:
                df.columns = ['x', 'y'] + list(df.columns[2:])

        coords = df[['x', 'y']].values.astype(np.float32)

        # ── FIXED GLOBAL NORMALIZATION ──────────────────────────────────────────
        # Uses dataset-wide constants computed once across all 600 training CSVs.
        # COORD_MIN = -108.5, COORD_MAX = 108.5, RANGE = 217.0 (both x and y).
        # coords_norm = (coords_raw - COORD_MIN) / COORD_RANGE * 2.0 - 1.0
        # This maps the full parametric bounding box to [-1, 1].
        c_min   = self.config.coord_min    # -108.5
        c_max   = self.config.coord_max    #  108.5
        c_range = c_max - c_min            #  217.0

        coords_norm = (coords - c_min) / c_range * 2.0 - 1.0

        transform_meta = {
            'orig_w':  orig_w,
            'orig_h':  orig_h,
            'c_min':   c_min,
            'c_max':   c_max,
            'c_range': c_range,
        }
        # ────────────────────────────────────────────────────────────────────────

        seq_len = min(len(coords_norm), self.config.max_seq_len)
        coords_norm = coords_norm[:seq_len]

        target_coords = np.zeros((self.config.max_seq_len, 2), dtype=np.float32)
        target_coords[:seq_len] = coords_norm

        pen_state = np.ones((self.config.max_seq_len,), dtype=np.float32)  # 1 = EOS / padding
        pen_state[:seq_len] = 0.0                                           # 0 = drawing

        pad_mask = np.ones((self.config.max_seq_len,), dtype=np.bool_)
        pad_mask[:seq_len] = False  # False = valid token for attention

        return {
            'image':        img_tensor,
            'target_coords': torch.tensor(target_coords),
            'pen_state':    torch.tensor(pen_state),
            'pad_mask':     torch.tensor(pad_mask),
            'seq_len':      torch.tensor(seq_len - 1, dtype=torch.long),
            'transform_meta': transform_meta,
        }


def denormalize(coords, transform_meta=None, config=None):
    """
    Invert fixed global normalization.
      coords_norm = (coords_raw - c_min) / c_range * 2.0 - 1.0
    =>coords_raw  = (coords_norm + 1.0) / 2.0 * c_range + c_min

    Pass either a transform_meta dict (from __getitem__) or a Config object.
    """
    if config is not None:
        c_min   = config.coord_min
        c_range = config.coord_max - config.coord_min
    elif transform_meta is not None:
        c_min   = transform_meta.get('c_min',   -108.5)
        c_range = transform_meta.get('c_range',  217.0)
    else:
        raise ValueError("Either transform_meta or config must be provided.")

    if isinstance(coords, torch.Tensor):
        out = coords.clone().float()
    else:
        out = coords.copy().astype(np.float32)

    out[..., 0] = (out[..., 0] + 1.0) / 2.0 * c_range + c_min
    out[..., 1] = (out[..., 1] + 1.0) / 2.0 * c_range + c_min
    return out


def collate_fn(batch):
    return {
        'image':         torch.stack([b['image']          for b in batch]),
        'target_coords': torch.stack([b['target_coords']  for b in batch]),
        'pen_state':     torch.stack([b['pen_state']      for b in batch]),
        'pad_mask':      torch.stack([b['pad_mask']       for b in batch]),
        'seq_len':       torch.stack([b['seq_len']        for b in batch]),
        'transform_meta': [b['transform_meta']            for b in batch],
    }
