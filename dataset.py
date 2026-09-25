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
        img = Image.open(img_path).convert('L') # Grayscale
        orig_w, orig_h = img.size
        img = img.resize((self.config.image_size, self.config.image_size), Image.BILINEAR)
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0 # (1, 224, 224)
        
        # Load coords
        csv_path = self.csv_paths[idx]
        df = pd.read_csv(csv_path)
        # Assume CSV has columns 'x', 'y'
        coords = df[['x', 'y']].values.astype(np.float32)
        
        # Normalization to [-1, 1] relative scales using image bounding box
        coords_norm = coords.copy()
        coords_norm[:, 0] = (coords[:, 0] / orig_w) * 2.0 - 1.0
        coords_norm[:, 1] = (coords[:, 1] / orig_h) * 2.0 - 1.0
        
        transform_meta = {
            'orig_w': orig_w,
            'orig_h': orig_h,
            'scale_x': orig_w / 2.0,
            'scale_y': orig_h / 2.0,
            'shift_x': orig_w / 2.0,
            'shift_y': orig_h / 2.0,
        }
        
        seq_len = min(len(coords_norm), self.config.max_seq_len)
        coords_norm = coords_norm[:seq_len]
        
        # Padding
        target_coords = np.zeros((self.config.max_seq_len, 2), dtype=np.float32)
        target_coords[:seq_len] = coords_norm
        
        pen_state = np.ones((self.config.max_seq_len,), dtype=np.float32) # 1 = EOS/Padding
        pen_state[:seq_len] = 0.0 # 0 = drawing
        
        pad_mask = np.ones((self.config.max_seq_len,), dtype=np.bool_)
        pad_mask[:seq_len] = False # False means not padded, valid for attention mask
        
        return {
            'image': img_tensor,
            'target_coords': torch.tensor(target_coords),
            'pen_state': torch.tensor(pen_state),
            'pad_mask': torch.tensor(pad_mask),
            'seq_len': torch.tensor(seq_len - 1, dtype=torch.long), # index of the last valid point
            'transform_meta': transform_meta
        }

def denormalize(coords, transform_meta):
    """
    Pure mapping function to denormalize coords using stored metadata.
    """
    scale_x = transform_meta['scale_x']
    scale_y = transform_meta['scale_y']
    shift_x = transform_meta['shift_x']
    shift_y = transform_meta['shift_y']
    
    if isinstance(coords, torch.Tensor):
        denorm_coords = coords.clone()
    else:
        denorm_coords = coords.copy()
        
    denorm_coords[..., 0] = (denorm_coords[..., 0] + 1.0) * scale_x
    denorm_coords[..., 1] = (denorm_coords[..., 1] + 1.0) * scale_y
    return denorm_coords

def collate_fn(batch):
    images = torch.stack([b['image'] for b in batch])
    target_coords = torch.stack([b['target_coords'] for b in batch])
    pen_state = torch.stack([b['pen_state'] for b in batch])
    pad_mask = torch.stack([b['pad_mask'] for b in batch])
    seq_lens = torch.stack([b['seq_len'] for b in batch])
    transform_metas = [b['transform_meta'] for b in batch]
    
    return {
        'image': images,
        'target_coords': target_coords,
        'pen_state': pen_state,
        'pad_mask': pad_mask,
        'seq_len': seq_lens,
        'transform_meta': transform_metas
    }
