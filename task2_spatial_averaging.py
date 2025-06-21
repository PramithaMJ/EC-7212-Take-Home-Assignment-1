"""
Task 2: Spatial Averaging
EC7212 - Computer Vision and Image Processing Assignment

This script performs spatial averaging on an image with different neighborhood sizes
(3x3, 10x10, and 20x20).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def spatial_averaging(image, kernel_size):
    """
    Perform spatial averaging on an image using a specified kernel size
    
    Args:
        image (ndarray): Input image
        kernel_size (int): Size of the averaging kernel (e.g., 3, 10, 20)
        
    Returns:
        ndarray: Spatially averaged image
    """
    # Create averaging kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    # Apply convolution
    averaged_image = cv2.filter2D(image, -1, kernel)
    
    return averaged_image.astype(np.uint8)

def display_results(images, titles, save_path=None):
    """
    Display multiple images in a grid
    
    Args:
        images (list): List of images to display
        titles (list): List of titles for each image
        save_path (str, optional): Path to save the figure
    """
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    # Define image options with paths relative to the 'images' directory
    image_options = {
        "lena": "Lenna_(test_image).png",  # Classic test image with good gradients
        "cameraman": "cameraman.tif",      # Good for showing edge preservation/blurring
        "mandrill": "mandrill.png",        # Highly textured image
        "moon": "moon.tif"                 # Good for showing crater details being smoothed
    }
    
    # Select the image to use
    selected_image = "lena"  # Cameraman is good for showing edge effects
    image_filename = image_options[selected_image]
    
    # Get absolute paths for images and results folders
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    results_dir = os.path.join(current_dir, "results")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Full path to the image file
    image_path = os.path.join(images_dir, image_filename)
    
    print(f"Attempting to load image from: {image_path}")
    
    # Load the image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Image loaded successfully: {original_image.shape}")
    
    # Task 2: Spatial averaging
    kernel_sizes = [3, 10, 20]
    averaging_images = [original_image]
    averaging_titles = ["Original"]
    
    for kernel_size in kernel_sizes:
        averaged = spatial_averaging(original_image, kernel_size)
        averaging_images.append(averaged)
        averaging_titles.append(f"{kernel_size}x{kernel_size} Average")
        print(f"Applied {kernel_size}x{kernel_size} spatial averaging")
    
    # Display and save results
    result_filename = f"task2_{selected_image}_spatial_averaging.png"
    result_path = os.path.join(results_dir, result_filename)
    display_results(averaging_images, averaging_titles, result_path)
    print(f"Results saved as {result_path}")

if __name__ == "__main__":
    main()
