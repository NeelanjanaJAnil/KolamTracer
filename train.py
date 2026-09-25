import os
import torch
import torch.optim as optim
import math
from torch.utils.data import DataLoader
from config import Config
from dataset import KolamPathDataset, collate_fn
from model import KolamModel
from loss import CompositeLoss

def pre_flight_verification(model, config):
    print("Executing pre-flight shape trace assertion...")
    B, T = 2, config.max_seq_len
    image = torch.randn(B, 1, config.image_size, config.image_size).to(config.device)
    target_coords = torch.randn(B, T, 2).to(config.device)
    pen_state = torch.randint(0, 2, (B, T)).float().to(config.device)
    pad_mask = torch.zeros((B, T), dtype=torch.bool).to(config.device)
    
    with torch.no_grad():
        delta_preds, eos_logits = model(image, target_coords, pen_state, pad_mask)
        assert delta_preds.shape == (B, T, 2), f"Expected {(B, T, 2)}, got {delta_preds.shape}"
        assert eos_logits.shape == (B, T, 1), f"Expected {(B, T, 1)}, got {eos_logits.shape}"
    print("Pre-flight shape trace successful.")

def get_lr_multiplier(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

def train():
    config = Config()
    
    # -------------------------------------------------------------
    # DATASET PATHS CONFIGURATION
    # The data_prep.py script automatically organizes files into 'data/images/' and 'data/csv/'
    # -------------------------------------------------------------
    base_dir = "data"
    images_dir = os.path.join(base_dir, "images")
    csv_dir = os.path.join(base_dir, "csv")
    
    image_paths = []
    csv_paths = []
    
    if os.path.exists(images_dir) and os.path.exists(csv_dir):
        for file_name in os.listdir(images_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_dir, file_name)
                csv_path = os.path.join(csv_dir, os.path.splitext(file_name)[0] + '.csv')
                if os.path.exists(csv_path):
                    image_paths.append(img_path)
                    csv_paths.append(csv_path)
                    
    print("Found {} valid image-CSV pairs for training.".format(len(image_paths)))
    # -------------------------------------------------------------
    
    train_dataset = KolamPathDataset(image_paths, csv_paths, config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=collate_fn, shuffle=True)
    
    model = KolamModel(config).to(config.device)
    criterion = CompositeLoss(config).to(config.device)
    
    pre_flight_verification(model, config)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.base_lr)
    
    global_step = 0
    total_steps = config.num_epochs * max(1, len(train_loader))
    
    print("Starting optimization loop...")
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            image = batch['image'].to(config.device)
            target_coords = batch['target_coords'].to(config.device)
            pen_state = batch['pen_state'].to(config.device)
            pad_mask = batch['pad_mask'].to(config.device)
            seq_len = batch['seq_len'].to(config.device)
            
            optimizer.zero_grad()
            
            # Scheduled sampling / Teacher forcing decay logic
            tf_ratio = max(
                config.teacher_forcing_end,
                config.teacher_forcing_start - (config.teacher_forcing_start - config.teacher_forcing_end) * (global_step / config.teacher_forcing_decay_steps)
            )
            
            delta_preds, eos_logits = model(image, target_coords, pen_state, pad_mask)
            loss, loss_dict = criterion(delta_preds, eos_logits, target_coords, pen_state, pad_mask, seq_len)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            
            # LR Scheduler mappings with cosine decay cycles
            lr_mult = get_lr_multiplier(global_step, config.warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.base_lr * lr_mult
                
            global_step += 1
            epoch_loss += loss.item()
            
        print("Epoch {}/{} - Loss: {:.4f}".format(epoch+1, config.num_epochs, epoch_loss/max(1, len(train_loader))))
        
        # Checkpointing including model weights, optimizer, and steps
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, "checkpoint_epoch_{}.pt".format(epoch))

if __name__ == '__main__':
    train()
