import torch
from config import Config
from model import KolamModel

def test_model():
    print("--- Execution Checkpoint: Forward Pass Shape Trace & Param Count ---\n")
    config = Config()
    model = KolamModel(config)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Trainable Parameters: {}".format(total_params))
    
    # Test forward pass shape trace
    B = 2
    T = config.max_seq_len
    
    # Dummy inputs
    image = torch.randn(B, 1, config.image_size, config.image_size)
    target_coords = torch.randn(B, T, 2)
    pen_state = torch.randint(0, 2, (B, T)).float()
    pad_mask = torch.zeros((B, T), dtype=torch.bool)
    
    print("\n[Input Shapes]")
    print(f"image:         {image.shape}")
    print(f"target_coords: {target_coords.shape}")
    print(f"pen_state:     {pen_state.shape}")
    print(f"pad_mask:      {pad_mask.shape}")
    
    try:
        delta_coords, eos_logits = model(image, target_coords, pen_state, pad_mask)
        print("\n[Output Shapes]")
        print(f"delta_coords:  {delta_coords.shape}")
        print(f"eos_logits:    {eos_logits.shape}")
        
        assert delta_coords.shape == (B, T, 2)
        assert eos_logits.shape == (B, T, 1)
        print("\n=> Forward pass shape trace verified successfully! Asserts passed.")
    except Exception as e:
        print("\n[Error during forward pass]")
        print(e)

if __name__ == '__main__':
    test_model()
