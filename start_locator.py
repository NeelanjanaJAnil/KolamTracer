import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

def find_start_coordinate(binary_image):
    """
    Extracts a starting coordinate from a raw Kolam binary image using skeleton endpoint detection.
    
    Convention: "topmost-then-leftmost skeleton endpoint"
    The Kaggle CSVs typically start the sequence at a specific logical grid point (often 1.0, 0.0
    in their parametric space). In image space, tracing usually begins at an endpoint of the curve.
    We detect all endpoints (pixels with exactly 1 neighbor) and pick the one with the smallest y, 
    then smallest x.
    """
    # 1. Skeletonize to 1-pixel width
    skeleton = skeletonize(binary_image > 0).astype(np.uint8)
    
    # 2. Find endpoints using a neighbor-counting filter
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]])
    
    # convolve the skeleton
    filtered = convolve(skeleton, kernel, mode='constant', cval=0)
    
    # An endpoint has exactly 1 neighbor, so the center (10) + 1 neighbor (1) = 11
    endpoints_y, endpoints_x = np.where(filtered == 11)
    
    if len(endpoints_y) == 0:
        # Fallback: if it's a perfectly closed loop, no endpoints exist. 
        # Just find the topmost-leftmost pixel of the skeleton.
        points_y, points_x = np.where(skeleton > 0)
        if len(points_y) == 0:
            return 0.0, 0.0
        
        # Sort by y (ascending), then x (ascending)
        sorted_indices = np.lexsort((points_x, points_y))
        return float(points_x[sorted_indices[0]]), float(points_y[sorted_indices[0]])
        
    # Sort endpoints by y, then x
    sorted_indices = np.lexsort((endpoints_x, endpoints_y))
    return float(endpoints_x[sorted_indices[0]]), float(endpoints_y[sorted_indices[0]])
