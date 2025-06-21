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
    print("\n=== Intensity Level Reduction ===\n")
    
    selected_image = "lena"
    image_filename = "lena_standard.png"
    
    max_level = 256
    
    print(f"Using image: lena_standard.png")
    
    while True:
        try:
            min_level = int(input("\nEnter desired intensity levels (e.g., 2, 4, 8... 256): "))
            if min_level > 0 and (min_level & (min_level - 1)) == 0 and min_level <= 256:
                break
            else:
                print("Please enter a positive power of 2 no greater than 256")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"Reducing intensity to {min_level} levels\n")
    
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
    
    levels_to_test = [min_level]
    print(f"Testing single intensity level: {min_level}")
    
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

if __name__ == "__main__":
    main()
