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
    if n_levels <= 0 or (n_levels & (n_levels - 1)) != 0:
        raise ValueError("Number of levels must be a positive power of 2")
    
    bits = int(np.log2(n_levels))
    
    reduced_image = (image >> (8 - bits)) << (8 - bits)
    
    return reduced_image

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
            safe_title = title.replace(' ', '_').replace('(', '').replace(')', '')
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
    
    if len(args) >= 1 and args[0] in image_options:
        interactive_mode = False
    else:
        interactive_mode = True
    
    if interactive_mode:
        print("\n=== Interactive Intensity Level Reduction ===\n")
        
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
        
        while True:
            try:
                desired_level = int(input("\nEnter desired intensity levels (e.g., 2, 4, 8... 256): "))
                if desired_level > 0 and (desired_level & (desired_level - 1)) == 0 and desired_level <= 256:
                    break
                else:
                    print("Please enter a positive power of 2 no greater than 256")
            except ValueError:
                print("Please enter a valid number")
        
        max_level = 256
        min_level = desired_level
        
        print(f"\nSelected image: {selected_image}")
        print(f"Reducing intensity to {desired_level} levels\n")
    else:
        # Default values
        selected_image = "jeep"
        max_level = 256
        min_level = 2
    
    if len(args) >= 1 and args[0] in image_options:
        selected_image = args[0]
        print(f"Using specified image: {selected_image}")
    
    if len(args) >= 3:
        try:
            max_level = int(args[1])
            min_level = int(args[2])
            
            max_level = min(max_level, 256)
            
            min_level = max(min_level, 2)
            
            print(f"Using specified levels: max={max_level}, min={min_level}")
        except ValueError:
            print("Invalid level values. Using defaults: max=256, min=2")
    
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
    
    def generate_intensity_levels(max_level=256, min_level=2):
        levels = []
        current_level = max_level // 2
        
        while current_level >= min_level:
            levels.append(current_level)
            current_level = current_level // 2
            
        return levels
    
    if interactive_mode and max_level > min_level:
        levels_to_test = [min_level]
        print(f"Testing single intensity level: {min_level}")
    else:
        levels_to_test = generate_intensity_levels(max_level, min_level)
        print(f"Testing intensity levels: {levels_to_test}")
    
    intensity_images = [original_image]
    intensity_titles = [f"Original ({max_level} levels)"]
    
    for level in levels_to_test:
        if not (level & (level - 1) == 0):
            print(f"Warning: {level} is not a power of 2, skipping...")
            continue
    
    for n_levels in levels_to_test:
        reduced = reduce_intensity_levels(original_image, n_levels)
        intensity_images.append(reduced)
        intensity_titles.append(f"{n_levels} levels")
        print(f"Reduced to {n_levels} intensity levels")
    
    task_dir = os.path.join(results_dir, f"task1_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    result_filename = f"task1_{selected_image}_intensity_reduction.png"
    result_path = os.path.join(results_dir, result_filename)
    
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
    print("       python3 task1_intensity_reduction.py")
    print("\nArguments:")
    print("  image_name    : Name of the image to use (lena, mandrill, smriti, jeep)")
    print("  max_level     : Maximum intensity level (default: 256)")
    print("  min_level     : Minimum intensity level (default: 2)")
    print("  [no arguments]: Interactive mode - prompts for image and specific intensity level")
    print("\nExample:")
    print("  python3 task1_intensity_reduction.py lena 256 2")
    print("  python3 task1_intensity_reduction.py")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
    else:
        main()
