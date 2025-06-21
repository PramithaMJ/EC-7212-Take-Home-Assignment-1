import cv2
import numpy as np
import matplotlib.pyplot as plt
from   scipy.ndimage import rotate
import os

class ImageProcessor:
    """
    A class to perform various image processing operations for EC7212 Assignment 1
    """
    
    def __init__(self, image_path):
        """
        Initialize with an image
        
        Args:
            image_path (str): Path to the input image
        """
        self.original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.image_path = image_path
        print(f"Image loaded successfully: {self.original_image.shape}")
    
    def intensity_level_reduction(self, num_levels):
        """
        Task 1: Reduce intensity levels from 256 to specified number (power of 2)
        
        Args:
            num_levels (int): Desired number of intensity levels (must be power of 2)
            
        Returns:
            numpy.ndarray: Image with reduced intensity levels
        """
        # Validate input
        if num_levels <= 0 or (num_levels & (num_levels - 1)) != 0:
            raise ValueError("Number of levels must be a positive power of 2")
        
        # Calculate the number of bits needed
        bits = int(np.log2(num_levels))
        
        # Reduce intensity levels
        # Shift right to reduce bits, then shift left to restore range
        reduced_image = (self.original_image >> (8 - bits)) << (8 - bits)
        
        return reduced_image
    
    def spatial_averaging(self, kernel_size):
        """
        Task 2: Perform spatial averaging with specified kernel size
        
        Args:
            kernel_size (int): Size of the averaging kernel (e.g., 3, 10, 20)
            
        Returns:
            numpy.ndarray: Spatially averaged image
        """
        # Create averaging kernel
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size * kernel_size)
        
        # Apply convolution
        averaged_image = cv2.filter2D(self.original_image, -1, kernel)
        
        return averaged_image.astype(np.uint8)
    
    def rotate_image(self, angle):
        """
        Task 3: Rotate image by specified angle
        
        Args:
            angle (float): Rotation angle in degrees
            
        Returns:
            numpy.ndarray: Rotated image
        """
        # Get image dimensions
        height, width = self.original_image.shape
        
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
        rotated_image = cv2.warpAffine(self.original_image, rotation_matrix, 
                                     (new_width, new_height), 
                                     borderValue=0)
        
        return rotated_image
    
    def block_averaging(self, block_size):
        """
        Task 4: Replace non-overlapping blocks with their average
        
        Args:
            block_size (int): Size of the blocks (e.g., 3, 5, 7)
            
        Returns:
            numpy.ndarray: Image with block averaging applied
        """
        height, width = self.original_image.shape
        result = np.copy(self.original_image).astype(np.float32)
        
        # Process blocks
        for i in range(0, height - block_size + 1, block_size):
            for j in range(0, width - block_size + 1, block_size):
                # Extract block
                block = self.original_image[i:i+block_size, j:j+block_size]
                
                # Calculate average
                avg_value = np.mean(block)
                
                # Replace all pixels in block with average
                result[i:i+block_size, j:j+block_size] = avg_value
        
        return result.astype(np.uint8)
    
    def display_results(self, images, titles, save_path=None):
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

def run_all_tasks():
    """
    Run all tasks and display results
    """
    # Choose one of the existing images in the workspace
    image_path = "Lenna_(test_image).png"  # Good for intensity level reduction
    
    # Alternative images available in the workspace:
    # image_path = "lena_standard.png"
    # image_path = "cameraman.tif" 
    # image_path = "mandrill.png"
    # image_path = "moon.tif"
    
    try:
        # Initialize processor
        processor = ImageProcessor(image_path)
        
        print("=" * 60)
        print("TASK 1: Intensity Level Reduction")
        print("=" * 60)
        
        # Task 1: Intensity level reduction
        levels_to_test = [2, 4, 8, 16, 32]
        intensity_images = [processor.original_image]
        intensity_titles = ["Original (256 levels)"]
        
        for levels in levels_to_test:
            reduced = processor.intensity_level_reduction(levels)
            intensity_images.append(reduced)
            intensity_titles.append(f"{levels} levels")
            print(f"Reduced to {levels} intensity levels")
        
        processor.display_results(intensity_images, intensity_titles, "task1_intensity_reduction.png")
        
        print("\n" + "=" * 60)
        print("TASK 2: Spatial Averaging")
        print("=" * 60)
        
        # Task 2: Spatial averaging
        kernel_sizes = [3, 10, 20]
        averaging_images = [processor.original_image]
        averaging_titles = ["Original"]
        
        for kernel_size in kernel_sizes:
            averaged = processor.spatial_averaging(kernel_size)
            averaging_images.append(averaged)
            averaging_titles.append(f"{kernel_size}x{kernel_size} Average")
            print(f"Applied {kernel_size}x{kernel_size} spatial averaging")
        
        processor.display_results(averaging_images, averaging_titles, "task2_spatial_averaging.png")
        
        print("\n" + "=" * 60)
        print("TASK 3: Image Rotation")
        print("=" * 60)
        
        # Task 3: Image rotation
        angles = [45, 90]
        rotation_images = [processor.original_image]
        rotation_titles = ["Original"]
        
        for angle in angles:
            rotated = processor.rotate_image(angle)
            rotation_images.append(rotated)
            rotation_titles.append(f"Rotated {angle}°")
            print(f"Rotated image by {angle} degrees")
        
        processor.display_results(rotation_images, rotation_titles, "task3_rotation.png")
        
        print("\n" + "=" * 60)
        print("TASK 4: Block Averaging")
        print("=" * 60)
        
        # Task 4: Block averaging
        block_sizes = [3, 5, 7]
        block_images = [processor.original_image]
        block_titles = ["Original"]
        
        for block_size in block_sizes:
            block_averaged = processor.block_averaging(block_size)
            block_images.append(block_averaged)
            block_titles.append(f"{block_size}x{block_size} Blocks")
            print(f"Applied {block_size}x{block_size} block averaging")
        
        processor.display_results(block_images, block_titles, "task4_block_averaging.png")
        
        print("\n" + "=" * 60)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Results saved as PNG files:")
        print("- task1_intensity_reduction.png")
        print("- task2_spatial_averaging.png")
        print("- task3_rotation.png")
        print("- task4_block_averaging.png")
        
    except Exception as e:
        print(f"Error: {e}")

def demonstrate_individual_functions():
    """
    Demonstrate how to use individual functions
    """
    print("\n" + "=" * 60)
    print("INDIVIDUAL FUNCTION DEMONSTRATIONS")
    print("=" * 60)
    
    # Use existing Lena image which is excellent for showing intensity reduction effects
    image_path = "Lenna_(test_image).png"
    
    processor = ImageProcessor(image_path)
    
    # Example 1: Intensity reduction to 4 levels
    reduced = processor.intensity_level_reduction(4)
    print(f"Original unique values: {len(np.unique(processor.original_image))}")
    print(f"Reduced unique values: {len(np.unique(reduced))}")
    
    # Example 2: 5x5 spatial averaging
    averaged = processor.spatial_averaging(5)
    print(f"Applied 5x5 spatial averaging")
    
    # Example 3: 30-degree rotation
    rotated = processor.rotate_image(30)
    print(f"Rotated by 30 degrees, new shape: {rotated.shape}")
    
    # Example 4: 4x4 block averaging
    block_avg = processor.block_averaging(4)
    print(f"Applied 4x4 block averaging")
    
    # No need to clean up since we're using an existing image

if __name__ == "__main__":
    # Run all tasks
    run_all_tasks()
    
    # Demonstrate individual functions
    demonstrate_individual_functions()