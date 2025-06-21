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

def display_results(images, titles, save_path=None, save_individual=False, individual_dir=None):
    """
    Display multiple images in a grid and optionally save individual images
    
    Args:
        images (list): List of images to display
        titles (list): List of titles for each image
        save_path (str, optional): Path to save the figure
        save_individual (bool, optional): Whether to save individual images
        individual_dir (str, optional): Directory to save individual images
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
        
        # Save individual images if requested
        if save_individual and individual_dir and i > 0:  # Skip original image (i=0)
            # Create safe filename from title
            safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('×', 'x')
            img_path = os.path.join(individual_dir, f"{safe_title}.png")
            
            # Create individual figure and save
            plt.figure(figsize=(5, 5))
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()  # Close individual figure
            print(f"Saved individual image: {img_path}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    """
    Main function to run block averaging
    Command line usage: python task4_block_averaging.py [image_name]
    """
    import sys
    
    # Define image options with paths relative to the 'images' directory
    image_options = {
        "lena": "lena_standard.png",      # Classic test image
        "mandrill": "mandrill.png",        # Highly detailed - great for showing resolution effects
        "smriti": "smriti.png",            # Additional test image
        "jeep": "jeep.png"                 # Additional test image

    }
    
    # Parse command line arguments if provided
    args = sys.argv[1:]
    
    # Default value
    selected_image = "jeep"
    
    # Process command line arguments if provided
    if len(args) >= 1 and args[0] in image_options:
        selected_image = args[0]
        print(f"Using specified image: {selected_image}")
    
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
    
    # Create a subdirectory for individual images
    task_dir = os.path.join(results_dir, f"task4_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    # Display and save results
    result_filename = f"task4_{selected_image}_block_averaging.png"
    result_path = os.path.join(results_dir, result_filename)
    
    # Save both the combined image and individual images
    display_results(
        block_images, 
        block_titles, 
        save_path=result_path,
        save_individual=True,
        individual_dir=task_dir
    )
    print(f"Results saved as {result_path}")
    print(f"Individual images saved in {task_dir}")

def print_usage():
    """Print usage instructions"""
    print("\nUsage: python3 task4_block_averaging.py [image_name]")
    print("\nArguments:")
    print("  image_name    : Name of the image to use (lena, mandrill, smriti)")
    print("\nExample:")
    print("  python3 task4_block_averaging.py lena")
    print("  python3 task4_block_averaging.py mandrill")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
