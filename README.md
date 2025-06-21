# Computer Vision and Image Processing - EC7212 Assignment

This project implements various image processing operations required for the EC7212 Computer Vision and Image Processing assignment.

## Project Structure

```
.
├── images/                 # Input images
│   ├── jeep.png
│   ├── lena_standard.png
│   ├── mandrill.png
│   └── smriti.png
├── results/                # Output images and individual task results
│   ├── task1_*/            # Task 1 individual results for each image
│   ├── task2_*/            # Task 2 individual results for each image
│   ├── task3_*/            # Task 3 individual results for each image
│   ├── task4_*/            # Task 4 individual results for each image
│   ├── task1_*_intensity_reduction.png
│   ├── task2_*_spatial_averaging.png
│   ├── task3_*_rotation.png
│   └── task4_*_block_averaging.png
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

### Task 1: Intensity Level Reduction

Task 1 runs in interactive mode by default, allowing you to select an image and specific intensity level:

```bash
# Run in interactive mode (default)
python3 task1_intensity_reduction.py
```

You can also run in command-line mode by specifying parameters:

```bash
# Run with command line arguments: image, max_level, min_level
python3 task1_intensity_reduction.py lena 256 2

# Other examples
python3 task1_intensity_reduction.py mandrill 256 4
python3 task1_intensity_reduction.py smriti 256 8
python3 task1_intensity_reduction.py jeep 256 16
```

### Task 2: Spatial Averaging

```bash
# Run with default image (jeep)
python3 task2_spatial_averaging.py

# Run with specific image
python3 task2_spatial_averaging.py lena
python3 task2_spatial_averaging.py mandrill
python3 task2_spatial_averaging.py smriti
```

### Task 3: Image Rotation

```bash
# Run with default image
python3 task3_image_rotation.py

# Run with specific image
python3 task3_image_rotation.py lena
python3 task3_image_rotation.py mandrill
python3 task3_image_rotation.py smriti
```

### Task 4: Block Averaging

```bash
# Run with default image
python3 task4_block_averaging.py

# Run with specific image
python3 task4_block_averaging.py lena
python3 task4_block_averaging.py mandrill
python3 task4_block_averaging.py smriti
```

### Getting Help

For more detailed usage instructions for any script:

```bash
python3 task1_intensity_reduction.py -h
python3 task2_spatial_averaging.py -h
python3 task3_image_rotation.py -h
python3 task4_block_averaging.py -h
```

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
- OpenCV (cv2)
- NumPy
- Matplotlib
