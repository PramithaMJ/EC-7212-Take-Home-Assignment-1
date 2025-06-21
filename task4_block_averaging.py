"""
Task 4: Spatial Resolution Reduction by Block Averaging
EC7212 - Computer Vision and Image Processing Assignment

This script reduces the spatial resolution of an image by replacing non-overlapping blocks
with their average values. Implemented for 3x3, 5x5, and 7x7 block sizes.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def block_averaging(image, block_size):
    """
    Replace non-overlapping blocks with their average
    
    Args:
        image (ndarray): Input image
        block_size (int): Size of the blocks (e.g., 3, 5, 7)
        
    Returns:
        ndarray: Image with block averaging applied
    """
    height, width = image.shape
    result = np.copy(image).astype(np.float32)
    
    # Process blocks
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            # Extract block
            block = image[i:i+block_size, j:j+block_size]
            
            # Calculate average
            avg_value = np.mean(block)
            
            # Replace all pixels in block with average
            result[i:i+block_size, j:j+block_size] = avg_value
    
    return result.astype(np.uint8)

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
        "lena": "Lenna_(test_image).png",  # Classic test image
        "cameraman": "cameraman.tif",      # Good for showing detail loss
        "mandrill": "mandrill.png",        # Highly detailed - great for showing resolution effects
        "moon": "moon.tif"                 # Good for showing crater details at lower resolution
    }
    
    # Select the image to use
    selected_image = "mandrill"
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
    
    # Task 4: Block averaging
    block_sizes = [3, 5, 7]
    block_images = [original_image]
    block_titles = ["Original"]
    
    for block_size in block_sizes:
        block_averaged = block_averaging(original_image, block_size)
        block_images.append(block_averaged)
        block_titles.append(f"{block_size}x{block_size} Blocks")
        print(f"Applied {block_size}x{block_size} block averaging")
    
    # Display and save results
    result_filename = f"task4_{selected_image}_block_averaging.png"
    result_path = os.path.join(results_dir, result_filename)
    display_results(block_images, block_titles, result_path)
    print(f"Results saved as {result_path}")

if __name__ == "__main__":
    main()
