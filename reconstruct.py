import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from dataset import denormalize

def filter_padding(coords, seq_len):
    """Filters out padding tokens utilizing seq_len to extract precise true tracking index lengths."""
    return coords[:seq_len + 1]

def reconstruct_b_spline(coords, transform_meta):
    """
    Parametric Cubic B-Splines explicitly using `per=1` to enforce
    seamless, closed periodic boundary conditions on one-stroke loops.
    """
    denorm_coords = denormalize(coords, transform_meta)
    
    x = denorm_coords[:, 0]
    y = denorm_coords[:, 1]
    
    # splprep requires per=1 for perfectly closed loops
    tck, u = interpolate.splprep([x, y], s=0, per=1)
    
    u_fine = np.linspace(0, 1, 1000)
    x_fine, y_fine = interpolate.splev(u_fine, tck)
    
    return x_fine, y_fine

def reconstruct_fourier_descriptors(coords, transform_meta, num_harmonics=10):
    """
    Parametric Fourier Descriptors calculating discrete Fourier coefficients 
    to generate analytical x(t) and y(t) sinusoid sums.
    """
    denorm_coords = denormalize(coords, transform_meta)
    
    x = denorm_coords[:, 0]
    y = denorm_coords[:, 1]
    N = len(x)
    
    # Complex coordinate representation
    z = x + 1j * y
    
    # Evaluate discrete Fourier transform
    Z = np.fft.fft(z)
    
    Z_filtered = np.zeros_like(Z)
    Z_filtered[:num_harmonics] = Z[:num_harmonics]
    if num_harmonics > 0:
        Z_filtered[-num_harmonics+1:] = Z[-num_harmonics+1:]
        
    # Generate analytical continuous sinusoid sums
    t_fine = np.linspace(0, 2*np.pi, 1000)
    x_fine = np.zeros_like(t_fine)
    y_fine = np.zeros_like(t_fine)
    
    for k in range(N):
        if Z_filtered[k] != 0:
            coeff = Z_filtered[k] / N
            # Extract harmonic frequency index accounting for Nyquist array folding
            n = k if k < N/2 else k - N
            x_fine += np.real(coeff * np.exp(1j * n * t_fine))
            y_fine += np.imag(coeff * np.exp(1j * n * t_fine))
            
    return x_fine, y_fine

def render_pathways(x_spline, y_spline, x_fourier, y_fourier):
    """
    Renders continuous mathematical vector pathways onto a digital canvas grid.
    """
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_spline, y_spline, 'b-', linewidth=2)
    plt.title('Parametric Cubic B-Splines (per=1)')
    plt.grid(True)
    plt.gca().invert_yaxis() # Match structural coordinate alignment
    
    plt.subplot(1, 2, 2)
    plt.plot(x_fourier, y_fourier, 'r-', linewidth=2)
    plt.title('Fourier Descriptors Analytical Sum')
    plt.grid(True)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('reconstruction_output.png')
    plt.close()
