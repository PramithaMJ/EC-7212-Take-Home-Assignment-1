import os
import requests
import numpy as np
import cv2

def download_image(url, filename):
    """Download an image from a URL and save it locally"""
    print(f"Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {filename}")
    else:
        print(f"Failed to download {filename}: HTTP {response.status_code}")

def main():
    """Download standard test images for computer vision tasks"""
    save_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Standard test images for computer vision
    images = {
        "lena.png": "https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png",
        "cameraman.tif": "https://raw.githubusercontent.com/rahuljain5/Image-Processing/master/cameraman.tif",
        "mandrill.png": "https://homepages.cae.wisc.edu/~ece533/images/baboon.png",
        "peppers.png": "https://homepages.cae.wisc.edu/~ece533/images/peppers.png",
        "moon.tif": "https://raw.githubusercontent.com/scikit-image/scikit-image/main/skimage/data/moon.png"
    }
    
    for filename, url in images.items():
        filepath = os.path.join(save_dir, filename)
        download_image(url, filepath)
    
    # Create a sample image with intensity gradients
    print("Creating intensity_test.png...")
    gradient = np.zeros((512, 512), dtype=np.uint8)
    for i in range(512):
        gradient[:, i] = i // 2  # Create horizontal gradient
    cv2.imwrite(os.path.join(save_dir, "intensity_test.png"), gradient)
    print("Successfully created intensity_test.png")
    
    print("\nAll test images have been downloaded to:")
    print(save_dir)
    print("\nAvailable images:")
    for filename in list(images.keys()) + ["intensity_test.png"]:
        print(f"- {filename}")

if __name__ == "__main__":
    main()
