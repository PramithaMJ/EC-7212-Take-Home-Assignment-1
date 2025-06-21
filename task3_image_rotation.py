"""
Task 3: Image Rotation
EC7212 - Computer Vision and Image Processing Assignment

This script rotates an image by 45 and 90 degrees.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rotate_image(image, angle):
    """
    Rotate image by specified angle
    
    Args:
        image (ndarray): Input image
        angle (float): Rotation angle in degrees
        
    Returns:
        ndarray: Rotated image
    """
    # Get image dimensions
    height, width = image.shape
    
    # Calculate rotation matrix
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new dimensions to avoid cropping
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int(height * sin_angle + width * cos_angle)
    new_height = int(height * cos_angle + width * sin_angle)
    
    # Adjust rotation matrix for new dimensions
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    
    # Apply rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, 
                                 (new_width, new_height), 
                                 borderValue=0)
    
    return rotated_image

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
        "cameraman": "cameraman.tif",      # Good for showing rotation effects
        "mandrill": "mandrill.png",        # Detailed image
        "moon": "moon.tif"                 # Circular object - interesting for rotation
    }
    
    # Select the image to use
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
    
    # Task 3: Image rotation
    angles = [45, 90]
    rotation_images = [original_image]
    rotation_titles = ["Original"]
    
    for angle in angles:
        rotated = rotate_image(original_image, angle)
        rotation_images.append(rotated)
        rotation_titles.append(f"Rotated {angle}°")
        print(f"Rotated image by {angle} degrees")
    
    # Display and save results
    result_filename = f"task3_{selected_image}_rotation.png"
    result_path = os.path.join(results_dir, result_filename)
    display_results(rotation_images, rotation_titles, result_path)
    print(f"Results saved as {result_path}")

if __name__ == "__main__":
    main()
