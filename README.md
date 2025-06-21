# Computer Vision and Image Processing - EC7212 Assignment

This project implements various image processing operations required for the EC7212 Computer Vision and Image Processing assignment.

## Project Structure

```
.
├── images/                 # Input images
│   └── lena_standard.png   # Main test image used by all scripts
├── results/                # Output images and individual task results
│   ├── task1_lena/         # Task 1 individual results (different intensity levels)
│   │   ├── 128_levels.png
│   │   ├── 64_levels.png
│   │   ├── 32_levels.png
│   │   ├── 16_levels.png
│   │   ├── 8_levels.png
│   │   ├── 4_levels.png
│   │   └── 2_levels.png
│   ├── task2_lena/         # Task 2 individual results (different kernel sizes)
│   │   ├── 3x3_Average.png
│   │   ├── 10x10_Average.png
│   │   └── 20x20_Average.png
│   ├── task3_lena/         # Task 3 individual results (different rotation angles)
│   │   ├── Rotated_45deg.png
│   │   └── Rotated_90deg.png
│   ├── task4_lena/         # Task 4 individual results (different block sizes)
│   │   ├── 3x3_Blocks.png
│   │   ├── 5x5_Blocks.png
│   │   └── 7x7_Blocks.png
│   ├── task1_lena_intensity_reduction.png  # Task 1 combined results
│   ├── task2_lena_spatial_averaging.png    # Task 2 combined results
│   ├── task3_lena_rotation.png             # Task 3 combined results
│   └── task4_lena_block_averaging.png      # Task 4 combined results
├── task1_intensity_reduction.py  # Task 1 implementation
├── task2_spatial_averaging.py    # Task 2 implementation
├── task3_image_rotation.py       # Task 3 implementation
├── task4_block_averaging.py      # Task 4 implementation
└── README.md                     # This file
```

## Task Descriptions

1. **Task 1: Intensity Level Reduction** (`task1_intensity_reduction.py`)
   - Reduces the number of intensity levels in an image from 256 to a specified power of 2
   - Tests with levels: 128, 64, 32, 16, 8, 4, 2
   - Features interactive mode for selecting specific intensity levels

2. **Task 2: Spatial Averaging** (`task2_spatial_averaging.py`)
   - Performs spatial averaging with different neighborhood sizes
   - Tests with kernel sizes: 3×3, 10×10, 20×20

3. **Task 3: Image Rotation** (`task3_image_rotation.py`)
   - Rotates an image by 45 and 90 degrees
   - Preserves all image content without cropping

4. **Task 4: Block Averaging** (`task4_block_averaging.py`)
   - Reduces spatial resolution by replacing non-overlapping blocks with their average
   - Tests with block sizes: 3×3, 5×5, 7×7

## Running the Scripts

All scripts are now simplified to use the "lena_standard.png" image by default. To run any of the tasks, simply execute the corresponding Python script:

### Task 1: Intensity Level Reduction

```bash
python3 task1_intensity_reduction.py
```

You'll be prompted to enter a desired intensity level (e.g., 2, 4, 8, 16, 32, 64, 128, or 256). The script will then reduce the Lena image to your specified number of intensity levels and display the results.

### Task 2: Spatial Averaging

```bash
python3 task2_spatial_averaging.py
```

This will apply spatial averaging to the Lena image with 3×3, 10×10, and 20×20 kernel sizes.

### Task 3: Image Rotation

```bash
python3 task3_image_rotation.py
```

This will rotate the Lena image by 45° and 90° angles.

### Task 4: Block Averaging

```bash
python3 task4_block_averaging.py
```

This will apply block averaging to the Lena image with 3×3, 5×5, and 7×7 block sizes.

## Available Images

The following test images are included in the project:
- `lena_standard.png`: Classic test image with good gradients
- `mandrill.png`: Highly textured image
- `smriti.png`: Additional test image
- `jeep.png`: Additional test image

## Output Results

When you run each script:
1. The processed images are displayed in a matplotlib window
2. A combined image showing all processing steps is saved in the `results/` directory
3. Individual processed images are saved in task-specific subdirectories (e.g., `results/task1_lena/`)

## Requirements

- Python 3.x
- OpenCV (cv2) - `pip install opencv-python`
- NumPy - `pip install numpy`
- Matplotlib - `pip install matplotlib`

## How to Install Dependencies

You can install all the required dependencies using pip:

```bash
pip install opencv-python numpy matplotlib
```

## Implementation Details

### Task 1: Intensity Level Reduction

This task reduces the number of intensity levels in an image from 256 (8-bit) to a desired power of 2 level (e.g., 128, 64, 32...). The implementation uses bit manipulation:

1. Calculate the number of bits needed for the desired intensity levels
2. Use bit shifting to remove the appropriate number of least significant bits
3. Display the original and reduced images side by side

### Task 2: Spatial Averaging

This task implements spatial averaging (mean filtering) with different kernel sizes:

1. Create averaging kernels of sizes 3×3, 10×10, and 20×20
2. Apply convolution using cv2.filter2D
3. Display results showing the effects of different kernel sizes

### Task 3: Image Rotation

This task rotates images by specific angles while preserving all image content:

1. Calculate the new image dimensions to fit the rotated image without cropping
2. Apply rotation transformation using cv2.warpAffine
3. Display original and rotated images

### Task 4: Block Averaging

This task reduces spatial resolution by replacing non-overlapping blocks with their average value:

1. Divide the image into non-overlapping blocks of specified size
2. Replace each block with the average intensity value of all pixels in that block
3. Display results showing the effects of different block sizes

- Python 3.x
- OpenCV (cv2)
- NumPy
- Matplotlib
