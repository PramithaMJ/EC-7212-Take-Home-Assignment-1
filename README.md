# Computer Vision and Image Processing - EC7212 Assignment

This project implements various image processing operations required for the EC7212 Computer Vision and Image Processing assignment.

## Project Structure

```
.
├── images/                 # Input images
│   ├── Lenna_(test_image).png
│   ├── cameraman.tif
│   ├── lena_standard.png
│   ├── mandrill.png
│   └── moon.tif
├── results/                # Output images
│   ├── task1_lena_intensity_reduction.png
│   ├── task2_cameraman_spatial_averaging.png
│   ├── task3_lena_rotation.png
│   └── task4_mandrill_block_averaging.png
├── task1_intensity_reduction.py
├── task2_spatial_averaging.py
├── task3_image_rotation.py
├── task4_block_averaging.py
└── README.md
```

## Task Descriptions

1. **Task 1: Intensity Level Reduction** (`task1_intensity_reduction.py`)
   - Reduces the number of intensity levels in an image from 256 to a specified power of 2
   - Tests with levels: 128, 64, 32, 16, 8, 4, 2

2. **Task 2: Spatial Averaging** (`task2_spatial_averaging.py`)
   - Performs spatial averaging with different neighborhood sizes
   - Tests with kernel sizes: 3×3, 10×10, 20×20

3. **Task 3: Image Rotation** (`task3_image_rotation.py`)
   - Rotates an image by 45 and 90 degrees
   - Preserves all image content without cropping

4. **Task 4: Block Averaging** (`task4_block_averaging.py`)
   - Reduces spatial resolution by replacing non-overlapping blocks with their average
   - Tests with block sizes: 3×3, 5×5, 7×7

## Running the Code

Each task is implemented as a separate Python script that can be run independently:

```bash
# Run Task 1: Intensity Level Reduction
python3 task1_intensity_reduction.py

# Run Task 2: Spatial Averaging
python3 task2_spatial_averaging.py

# Run Task 3: Image Rotation
python3 task3_image_rotation.py

# Run Task 4: Block Averaging
python3 task4_block_averaging.py
```

## Changing Test Images

Each script comes with a selection of test images. To use a different image, modify the `selected_image` variable in the `main()` function of each script:

```python
# Available options: "lena", "cameraman", "mandrill", "moon"
selected_image = "lena"  # Change this to use a different image
```

## Requirements

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
