import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.morphology import skeletonize
from skimage import filters
import io
import os

from config import Config
from model import KolamModel
from dataset import denormalize
from reconstruct import reconstruct_b_spline, reconstruct_fourier_descriptors

def get_fourier_latex(coords, transform_meta, num_harmonics=10):
    denorm_coords = denormalize(coords, transform_meta)
    x = denorm_coords[:, 0]
    y = denorm_coords[:, 1]
    N = len(x)
    z = x + 1j * y
    Z = np.fft.fft(z)
    
    x_terms = []
    y_terms = []
    
    for k in range(N):
        if k <= num_harmonics or k >= N - num_harmonics:
            coeff = Z[k] / N
            if abs(coeff) < 1e-4: continue
            n = k if k < N/2 else k - N
            A = np.real(coeff)
            B = np.imag(coeff)
            
            if n == 0:
                x_terms.append("{:.2f}".format(A))
                y_terms.append("{:.2f}".format(B))
            else:
                x_terms.append("({:.2f}\\cos({}t) - {:.2f}\\sin({}t))".format(A, n, B, n))
                y_terms.append("({:.2f}\\sin({}t) + {:.2f}\\cos({}t))".format(A, n, B, n))
                
    if not x_terms: x_terms = ["0"]
    if not y_terms: y_terms = ["0"]
    
    x_latex = "$$x(t) = " + " + ".join(x_terms).replace("+ -", "- ") + "$$"
    y_latex = "$$y(t) = " + " + ".join(y_terms).replace("+ -", "- ") + "$$"
    return x_latex + "\n\n" + y_latex

def preprocess_image(img):
    img_gray = img.convert('L')
    config = Config()
    orig_w, orig_h = img.size
    img_resized = img_gray.resize((config.image_size, config.image_size), Image.BILINEAR)
    img_np = np.array(img_resized)
    thresh = filters.threshold_otsu(img_np)
    binary = img_np < thresh 
    skeleton = skeletonize(binary)
    skeleton_img = (skeleton * 255).astype(np.uint8)
    
    img_tensor = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
    
    transform_meta = {
        'orig_w': float(orig_w),
        'orig_h': float(orig_h),
        'scale_x': float(orig_w) / 2.0,
        'scale_y': float(orig_h) / 2.0,
        'shift_x': float(orig_w) / 2.0,
        'shift_y': float(orig_h) / 2.0,
    }
    return img_tensor, skeleton_img, transform_meta

def main():
    st.set_page_config(layout="wide", page_title="Kolam Tracer Engine")
    
    st.title("🎨 Universal Kolam Tracer - Sequence-to-Sequence Autoregressive Engine")
    st.markdown("Upload a hand-drawn Kolam image. The structural engine will automatically extract the skeletal logic and execute an autoregressive delta-step trace.")
    
    uploaded_file = st.file_uploader("Upload Kolam Image (PNG/JPG)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded Kolam", use_container_width=True)
            
        with st.spinner("Processing trace pipeline..."):
            config = Config()
            img_tensor, skeleton_img, transform_meta = preprocess_image(image)
            
            model = KolamModel(config)
            if os.path.exists("checkpoint_epoch_0.pt"):
                try:
                    checkpoint = torch.load("checkpoint_epoch_0.pt", map_location="cpu")
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e:
                    st.warning("Failed to load checkpoint, utilizing random weights: {}".format(e))
            
            model.eval()
            with torch.no_grad():
                memory = model.encoder(img_tensor)
                pred_coords, pred_pen = model.decoder.autoregressive_generate(memory, config.max_seq_len)
            
            coords = pred_coords[0].cpu().numpy()
            
            x_spline, y_spline = reconstruct_b_spline(coords, transform_meta)
            x_fourier, y_fourier = reconstruct_fourier_descriptors(coords, transform_meta, num_harmonics=10)
            latex_formulas = get_fourier_latex(coords, transform_meta, num_harmonics=10)
            
            with col2:
                st.markdown("### Fourier Algebraic Equations")
                st.markdown(latex_formulas)
                
            st.markdown("---")
            st.markdown("### Reconstruction Canvas: 3-Way Visual Comparison")
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            axes[0].imshow(skeleton_img, cmap='gray')
            axes[0].set_title("Stage 1/2: Thinned Skeleton Mask")
            axes[0].axis('off')
            
            axes[1].plot(x_spline, y_spline, 'b-', linewidth=2.5)
            axes[1].set_title("B-Spline Closed Loop Curve")
            axes[1].invert_yaxis()
            axes[1].grid(True)
            
            axes[2].plot(x_fourier, y_fourier, 'r-', linewidth=2.5)
            axes[2].set_title("Fourier Descriptor Harmonic Waveforms")
            axes[2].invert_yaxis()
            axes[2].grid(True)
            
            plt.tight_layout()
            st.pyplot(fig)

if __name__ == "__main__":
    main()
