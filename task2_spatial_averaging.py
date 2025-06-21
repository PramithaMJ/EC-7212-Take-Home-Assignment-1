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
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
    
    averaged_image = cv2.filter2D(image, -1, kernel)
    
    return averaged_image.astype(np.uint8)

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
            safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('×', 'x')
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
    image_options = {
        "lena": "lena_standard.png",
        "mandrill": "mandrill.png",
        "smriti": "smriti.png",
        "jeep": "jeep.png"
    }
    
    print("\n=== Spatial Averaging ===\n")
    
    print("Available images:")
    for i, (name, _) in enumerate(image_options.items(), 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            img_choice = int(input("\nSelect image number: "))
            if 1 <= img_choice <= len(image_options):
                selected_image = list(image_options.keys())[img_choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(image_options)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nSelected image: {selected_image}\n")
    
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
    
    kernel_sizes = [3, 10, 20]
    averaging_images = [original_image]
    averaging_titles = ["Original"]
    
    for kernel_size in kernel_sizes:
        averaged = spatial_averaging(original_image, kernel_size)
        averaging_images.append(averaged)
        averaging_titles.append(f"{kernel_size}x{kernel_size} Average")
        print(f"Applied {kernel_size}x{kernel_size} spatial averaging")
    
    task_dir = os.path.join(results_dir, f"task2_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    result_filename = f"task2_{selected_image}_spatial_averaging.png"
    result_path = os.path.join(results_dir, result_filename)
    
    display_results(
        averaging_images, 
        averaging_titles, 
        save_path=result_path,
        save_individual=True,
        individual_dir=task_dir
    )
    print(f"Results saved as {result_path}")
    print(f"Individual images saved in {task_dir}")

if __name__ == "__main__":
    main()
