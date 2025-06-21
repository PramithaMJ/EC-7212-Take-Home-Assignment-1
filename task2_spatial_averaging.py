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
    Main function to run spatial averaging
    Command line usage: python task2_spatial_averaging.py [image_name]
    """
    import sys
    
    # Define image options with paths relative to the 'images' directory
    image_options = {
        "lena": "lena_standard.png",       # Classic test image with good gradients
        "mandrill": "mandrill.png",        # Highly textured image
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
    
    # Task 2: Spatial averaging
    kernel_sizes = [3, 10, 20]
    averaging_images = [original_image]
    averaging_titles = ["Original"]
    
    for kernel_size in kernel_sizes:
        averaged = spatial_averaging(original_image, kernel_size)
        averaging_images.append(averaged)
        averaging_titles.append(f"{kernel_size}x{kernel_size} Average")
        print(f"Applied {kernel_size}x{kernel_size} spatial averaging")
    
    # Create a subdirectory for individual images
    task_dir = os.path.join(results_dir, f"task2_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    # Display and save results
    result_filename = f"task2_{selected_image}_spatial_averaging.png"
    result_path = os.path.join(results_dir, result_filename)
    
    # Save both the combined image and individual images
    display_results(
        averaging_images, 
        averaging_titles, 
        save_path=result_path,
        save_individual=True,
        individual_dir=task_dir
    )
    print(f"Results saved as {result_path}")
    print(f"Individual images saved in {task_dir}")

def print_usage():
    """Print usage instructions"""
    print("\nUsage: python3 task2_spatial_averaging.py [image_name]")
    print("\nArguments:")
    print("  image_name    : Name of the image to use (lena, mandrill, smriti)")
    print("\nExample:")
    print("  python3 task2_spatial_averaging.py lena")
    print("  python3 task2_spatial_averaging.py mandrill")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
