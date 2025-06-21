import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def block_averaging(image, block_size):
    height, width = image.shape
    result = np.copy(image).astype(np.float32)
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            block = image[i:i+block_size, j:j+block_size]
            
            avg_value = np.mean(block)
            
            result[i:i+block_size, j:j+block_size] = avg_value
    
    return result.astype(np.uint8)

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
    print("\n=== Block Averaging Resolution Reduction ===\n")
    
    selected_image = "lena"
    image_filename = "lena_standard.png"
    
    print(f"Using image: lena_standard.png\n")
    
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
    
    block_sizes = [3, 5, 7]
    block_images = [original_image]
    block_titles = ["Original"]
    
    for block_size in block_sizes:
        block_averaged = block_averaging(original_image, block_size)
        block_images.append(block_averaged)
        block_titles.append(f"{block_size}x{block_size} Blocks")
        print(f"Applied {block_size}x{block_size} block averaging")
    
    task_dir = os.path.join(results_dir, f"task4_{selected_image}")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
        print(f"Created directory for individual images: {task_dir}")
    
    result_filename = f"task4_{selected_image}_block_averaging.png"
    result_path = os.path.join(results_dir, result_filename)
    
    display_results(
        block_images, 
        block_titles, 
        save_path=result_path,
        save_individual=True,
        individual_dir=task_dir
    )
    print(f"Results saved as {result_path}")
    print(f"Individual images saved in {task_dir}")

if __name__ == "__main__":
    main()
