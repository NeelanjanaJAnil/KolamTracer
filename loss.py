import torch
import torch.nn as nn
import torch.nn.functional as F

class CompositeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.coord_loss_type == "huber":
            self.coord_criterion = nn.SmoothL1Loss(beta=config.huber_beta, reduction='none')
        else:
            self.coord_criterion = nn.MSELoss(reduction='none')
            
        self.pen_criterion = nn.BCEWithLogitsLoss(reduction='none')
        
    def forward(self, delta_preds, eos_logits, target_coords, target_pen, pad_mask, seq_len):
        B, T, _ = delta_preds.shape
        
        # Calculate precise per-step delta targets (not absolute coordinates)
        delta_targets = torch.zeros_like(target_coords)
        delta_targets[:, :-1, :] = target_coords[:, 1:, :] - target_coords[:, :-1, :]
        
        # Primary coord loss (masked)
        valid_mask = ~pad_mask # (B, T)
        
        # Delta loss is only valid up to seq_len - 1
        delta_valid_mask = valid_mask.clone()
        for b in range(B):
            delta_valid_mask[b, seq_len[b]:] = False
            
        # Isolate per-step Delta-targets precisely using the calculated mask
        coord_loss = self.coord_criterion(delta_preds, delta_targets)
        coord_loss = (coord_loss.sum(dim=-1) * delta_valid_mask.float()).sum() / (delta_valid_mask.float().sum() + 1e-8)
        
        # Pen State Loss (Masked BCE)
        eos_loss = self.pen_criterion(eos_logits.squeeze(-1), target_pen.float())
        eos_loss = (eos_loss * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8)
        
        # Auxiliary Curvature Loss: penalizing large second-differences
        second_diff = delta_preds[:, 1:, :] - delta_preds[:, :-1, :]
        curv_valid_mask = delta_valid_mask[:, 1:]
        curv_loss = (second_diff.pow(2).sum(dim=-1) * curv_valid_mask.float()).sum() / (curv_valid_mask.float().sum() + 1e-8)
        
        # ClosedLoop Loss: minimizes L2 distance between final valid index and the origin point 
        # (calculated on accumulated absolute positions)
        pred_abs_coords = target_coords[:, 0:1, :] + torch.cumsum(delta_preds, dim=1)
        closed_loop_loss = 0.0
        for b in range(B):
            start_p = target_coords[b, 0] # Origin point
            end_p = pred_abs_coords[b, max(0, seq_len[b] - 1)] # Final valid position
            closed_loop_loss += F.mse_loss(end_p, start_p)
        closed_loop_loss = closed_loop_loss / B
        
        total_loss = coord_loss + 0.1 * eos_loss + 0.01 * curv_loss + 0.05 * closed_loop_loss
        
        return total_loss, {
            'coord_loss': coord_loss.item(),
            'eos_loss': eos_loss.item(),
            'curv_loss': curv_loss.item(),
            'closed_loop_loss': closed_loop_loss.item()
        }
