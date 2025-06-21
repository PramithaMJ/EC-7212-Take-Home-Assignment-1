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

def reduce_intensity_levels(image, n_levels):
    """
    Reduce intensity levels from 256 to specified number (power of 2)
    
    Args:
        image (ndarray): Input image
        n_levels (int): Desired number of intensity levels (must be power of 2)
        
    Returns:
        ndarray: Image with reduced intensity levels
    """
    # Validate input
    if n_levels <= 0 or (n_levels & (n_levels - 1)) != 0:
        raise ValueError("Number of levels must be a positive power of 2")
    
    # Calculate the number of bits needed
    bits = int(np.log2(n_levels))
    
    # Reduce intensity levels
    # Shift right to reduce bits, then shift left to restore range
    reduced_image = (image >> (8 - bits)) << (8 - bits)
    
    return reduced_image

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
            safe_title = title.replace(' ', '_').replace('(', '').replace(')', '')
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
    Main function to run the intensity level reduction
    Command line usage: python task1_intensity_reduction.py [image_name] [max_level] [min_level]
    """
    import sys
    
    # Define image options with paths relative to the 'images' directory
    image_options = {
        "lena": "lena_standard.png",      # Classic test image with good gradients
        "mandrill": "mandrill.png",        # Detailed texture
        "smriti": "smriti.png"             # Additional test image
    }
    
    # Parse command line arguments if provided
    args = sys.argv[1:]
    
    # Default values
    selected_image = "mandrill"
    max_level = 256
    min_level = 2
    
    # Process command line arguments if provided
    if len(args) >= 1 and args[0] in image_options:
        selected_image = args[0]
        print(f"Using specified image: {selected_image}")
    
    if len(args) >= 3:
        try:
            max_level = int(args[1])
            min_level = int(args[2])
            
            # Ensure max_level is 256 or less (8-bit images)
            max_level = min(max_level, 256)
            
            # Ensure min_level is at least 2
            min_level = max(min_level, 2)
            
            print(f"Using specified levels: max={max_level}, min={min_level}")
        except ValueError:
            print("Invalid level values. Using defaults: max=256, min=2")
    
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
    
    # Function to generate intensity levels from max_level down to min_level in powers of 2
    def generate_intensity_levels(max_level=256, min_level=2):
        """
        Generate a list of intensity levels in powers of 2, from max_level down to min_level
        
        Args:
            max_level (int): Maximum intensity level (default: 256)
            min_level (int): Minimum intensity level (default: 2)
            
        Returns:
            list: List of intensity levels in descending order
        """
        levels = []
        current_level = max_level // 2  # Start from max_level/2 since max_level is the original
        
        while current_level >= min_level:
            levels.append(current_level)
            current_level = current_level // 2
            
        return levels
    
    # Intensity level reduction - use the max_level and min_level from command line arguments
    # Note: These values are already set from command line arguments or defaults
    
    # Generate levels dynamically instead of hardcoding
    levels_to_test = generate_intensity_levels(max_level, min_level)
    print(f"Testing intensity levels: {levels_to_test}")
    
    intensity_images = [original_image]
    intensity_titles = [f"Original ({max_level} levels)"]
    
    # Validate intensity levels
    for level in levels_to_test:
        if not (level & (level - 1) == 0):  # Check if power of 2
            print(f"Warning: {level} is not a power of 2, skipping...")
            continue
    
    for n_levels in levels_to_test:
        reduced = reduce_intensity_levels(original_image, n_levels)
        intensity_images.append(reduced)
        intensity_titles.append(f"{n_levels} levels")
        print(f"Reduced to {n_levels} intensity levels")
    
    # Create a subdirectory for individual images
    task_dir = os.path.join(results_dir, f"task1_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    # Display and save results
    result_filename = f"task1_{selected_image}_intensity_reduction.png"
    result_path = os.path.join(results_dir, result_filename)
    
    # Save both the combined image and individual images
    display_results(
        intensity_images, 
        intensity_titles, 
        save_path=result_path,
        save_individual=True,
        individual_dir=task_dir
    )
    print(f"Results saved as {result_path}")
    print(f"Individual images saved in {task_dir}")

def print_usage():
    """Print usage instructions"""
    print("\nUsage: python3 task1_intensity_reduction.py [image_name] [max_level] [min_level]")
    print("\nArguments:")
    print("  image_name    : Name of the image to use (lena, mandrill, smriti)")
    print("  max_level     : Maximum intensity level (default: 256)")
    print("  min_level     : Minimum intensity level (default: 2)")
    print("\nExample:")
    print("  python3 task1_intensity_reduction.py lena 256 2")
    print("  python3 task1_intensity_reduction.py mandrill 256 2")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
