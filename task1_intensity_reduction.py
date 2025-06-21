"""
Task 1: Intensity Level Reduction
EC7212 - Computer Vision and Image Processing Assignment

This script reduces the number of intensity levels in an image from 256 to a specified number
(which must be a power of 2).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def reduce_intensity_levels(image, num_levels):
    """
    Reduce intensity levels from 256 to specified number (power of 2)
    
    Args:
        image (ndarray): Input image
        num_levels (int): Desired number of intensity levels (must be power of 2)
        
    Returns:
        ndarray: Image with reduced intensity levels
    """
    # Validate input
    if num_levels <= 0 or (num_levels & (num_levels - 1)) != 0:
        raise ValueError("Number of levels must be a positive power of 2")
    
    # Calculate the number of bits needed
    bits = int(np.log2(num_levels))
    
    # Reduce intensity levels
    # Shift right to reduce bits, then shift left to restore range
    reduced_image = (image >> (8 - bits)) << (8 - bits)
    
    return reduced_image

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
        "cameraman": "cameraman.tif",      # Good contrast
        "mandrill": "mandrill.png",        # Detailed texture
        "moon": "moon.tif"                 # Grayscale with interesting features
    }
    
    # Select the image to use - change this to use a different image
    selected_image = "lena"
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
    
    # Intensity level reduction
    levels_to_test = [128, 64, 32, 16, 8, 4, 2]
    intensity_images = [original_image]
    intensity_titles = ["Original (256 levels)"]
    
    for levels in levels_to_test:
        reduced = reduce_intensity_levels(original_image, levels)
        intensity_images.append(reduced)
        intensity_titles.append(f"{levels} levels")
        print(f"Reduced to {levels} intensity levels")
    
    # Display and save results
    result_filename = f"task1_{selected_image}_intensity_reduction.png"
    result_path = os.path.join(results_dir, result_filename)
    display_results(intensity_images, intensity_titles, result_path)
    print(f"Results saved as {result_path}")

if __name__ == "__main__":
    main()
