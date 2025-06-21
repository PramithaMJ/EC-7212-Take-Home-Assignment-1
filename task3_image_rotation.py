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
    height, width = image.shape
    
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    cos_angle = np.abs(rotation_matrix[0, 0])
    sin_angle = np.abs(rotation_matrix[0, 1])
    new_width = int(height * sin_angle + width * cos_angle)
    new_height = int(height * cos_angle + width * sin_angle)
    
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2
    
    rotated_image = cv2.warpAffine(image, rotation_matrix, 
                                 (new_width, new_height), 
                                 borderValue=0)
    
    return rotated_image

def display_results(images, titles, save_path=None, save_individual=False, individual_dir=None):
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        
        if save_individual and individual_dir and i > 0:
            safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('°', 'deg')
            img_path = os.path.join(individual_dir, f"{safe_title}.png")
            
            plt.figure(figsize=(5, 5))
            plt.imshow(img, cmap='gray')
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved individual image: {img_path}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def main():
    import sys
    
    image_options = {
        "lena": "lena_standard.png",
        "mandrill": "mandrill.png",
        "smriti": "smriti.png",
        "jeep": "jeep.png" 
    }
    
    args = sys.argv[1:]
    
    selected_image = "jeep"
    
    if len(args) >= 1 and args[0] in image_options:
        selected_image = args[0]
        print(f"Using specified image: {selected_image}")
    
    image_filename = image_options[selected_image]
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(current_dir, "images")
    results_dir = os.path.join(current_dir, "results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    image_path = os.path.join(images_dir, image_filename)
    
    print(f"Attempting to load image from: {image_path}")
    
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    print(f"Image loaded successfully: {original_image.shape}")
    
    angles = [45, 90]
    rotation_images = [original_image]
    rotation_titles = ["Original"]
    
    for angle in angles:
        rotated = rotate_image(original_image, angle)
        rotation_images.append(rotated)
        rotation_titles.append(f"Rotated {angle}°")
        print(f"Rotated image by {angle} degrees")
    
    task_dir = os.path.join(results_dir, f"task3_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    result_filename = f"task3_{selected_image}_rotation.png"
    result_path = os.path.join(results_dir, result_filename)
    
    display_results(
        rotation_images, 
        rotation_titles, 
        save_path=result_path,
        save_individual=True,
        individual_dir=task_dir
    )
    print(f"Results saved as {result_path}")
    print(f"Individual images saved in {task_dir}")

def print_usage():
    """Print usage instructions"""
    print("\nUsage: python3 task3_image_rotation.py [image_name]")
    print("\nArguments:")
    print("  image_name    : Name of the image to use (lena, mandrill, smriti)")
    print("\nExample:")
    print("  python3 task3_image_rotation.py lena")
    print("  python3 task3_image_rotation.py mandrill")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
