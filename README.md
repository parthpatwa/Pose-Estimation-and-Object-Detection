# Pose Estimation with YOLOv7

This repository contains the implementation of a pose estimation system using the YOLOv7 model. It is capable of detecting human figures in images and estimating their pose by identifying and 
plotting keypoints on the human body.
[model link](https://drive.google.com/file/d/1QPiC2lBpKj9UlYTMUbf-2ABatqUhOHER/view?usp=sharing)

## Features

- Pose estimation with state-of-the-art YOLOv7 model.
- Customizable input for model weights and input frame.
- Ability to run on CPU or GPU.
- Options to view results, save confidence scores, and adjust line thickness.
- Hides labels and confidences if desired.

## Prerequisites

- Python 3.6+
- PyTorch 1.7+
- OpenCV Library
- Torchvision

## Installation

Clone the repository to your local machine:

```sh
git clone [repository-url]
cd [repository-name]
```

Install the required dependencies:

```sh
pip install -r requirements.txt
```

## Usage

To run the pose estimation, execute the `pose-estimate.py` script with the necessary arguments:

```sh
python pose-estimate.py --poseweights [path-to-weights] --frame [path-to-frame]
```

Optional arguments include:

- `--device`: Specify 'cpu' or GPU device indices for inference.
- `--view-img`: Set to true to display results.
- `--save-conf`: Set to true to save confidence scores.
- `--line-thickness`: Set line thickness for bounding boxes (in pixels).
- `--hide-labels`: Set to true to hide labels in the output image.
- `--hide-conf`: Set to true to hide confidence scores in the output image.

## Output

The script will produce an image with detected poses, saved as `result.jpg` in the current directory.

### `letterbox` Function

The `letterbox` function is a common utility in object detection and pose estimation tasks. Its primary purpose is to resize an image to a new shape while maintaining the aspect ratio. It does this by adding padding of a certain color (usually gray) if necessary.

Here's a step-by-step breakdown of what the `letterbox` function does:

1. **Define New Shape**: It determines the new shape of the image, which is either specified by the user or set to a default size.

2. **Calculate Scale Ratio**: The function calculates the scale ratio to resize the image such that the aspect ratio is maintained. It ensures that either the width or the height of the image matches the new dimensions while the other dimension is scaled appropriately.

3. **Padding Calculation**: If there's a need to add padding to maintain the aspect ratio, the function calculates how much padding is needed on each side of the image.

4. **Resizing and Padding**: The image is resized to the new dimensions, and if required, padding is added to the top, bottom, left, and right sides of the image. This is done using OpenCV's `copyMakeBorder` function.

5. **Return**: The function returns the resized and padded image, the ratio of the new size to the old size, and the padding that was added to each side.

This process is crucial because most neural networks require the input to be of a fixed size. The `letterbox` function ensures that the images fed into the model are consistent in size without distorting the original image content.

### `PoseDetector` Class

The `PoseDetector` class is designed to encapsulate the functionality needed to perform pose detection using a pre-trained YOLOv7 model. Here's what each part of the class does:

- **Initialization (`__init__`)**: The constructor initializes the class with various parameters like the weights file for the pose model, the image frame to process, and various flags that control the behavior of the detection (like whether to view the image, save confidence levels, etc.).

- **Device Selection**: It uses the utility function `select_device` to set up the processing device (CPU or GPU).

- **Model Loading**: The pre-trained model weights are loaded into the model architecture using `attempt_load`.

- **Image Processing (`run`)**: The `run` method is where the main processing happens. It reads the input image, resizes it, converts it to a tensor, and then passes it through the model to detect poses. The pose data is then used to plot keypoints on the original image.

- **Keypoint and Box Plotting**: For each detected pose, the function `plot_one_box_kpt` is called to draw the bounding boxes and keypoints on the image.

- **Saving Output**: The processed image with the poses plotted is saved to a file named 'result.jpg'.

Overall, the `PoseDetector` class abstracts the complexity of pose detection into a simple-to-use interface that can be readily used with different images and model weights.

----------------------------

Some potential limitations, areas for improvement, ways to optimize inference speed, and general difficulties that one might encounter when implementing such a solution. 

### Limitations:

1. **Hardware Constraints**: The code defaults to CPU for inference, which is considerably slower than GPU. Running advanced models like YOLOv7 on CPUs may not be practical for real-time applications.

2. **Model Generalization**: The performance of the pose estimation model may degrade if it encounters poses or scenarios that were underrepresented in the training data.

3. **Single Image Inference**: The provided code is designed for single image inference, which may not be suitable for real-time video stream processing where batch processing could be more efficient.

4. **Resolution Dependence**: The resizing of images to a fixed input size can lead to loss of detail, especially for smaller objects within an image, which can affect pose estimation accuracy.

5. **Fixed Thresholds**: The use of fixed thresholds for non-max suppression and confidence scores may not be optimal for all scenarios and could be improved by making them adaptable to the image content.

### Potential Improvements:

1. **Dynamic Scaling**: Implementing adaptive scaling methods that consider the content of the image could help preserve important details during the resizing process.

2. **Batch Processing**: Modifying the code to process images in batches can significantly improve throughput, especially on GPUs.

3. **Model Quantization**: Applying model quantization techniques could reduce the model size and speed up inference without significant loss of accuracy.

4. **Optimized Libraries**: Using optimized versions of PyTorch and OpenCV built for specific hardware can yield better performance.

5. **Augmented Datasets**: Training the model on a more diverse dataset can improve its robustness and accuracy in various real-world scenarios.

### Optimizing Inference Speed:

1. **Model Pruning**: Pruning the model to remove redundancy can lead to faster inference times.

2. **Reduced Precision Inference**: Using reduced precision (e.g., FP16 or INT8) can speed up model inference on compatible hardware.

3. **Efficient Image Pre-processing**: Optimizing the `letterbox` function to minimize the use of padding or using a more efficient resizing algorithm could reduce pre-processing time.

4. **Parallel Processing**: Implementing multi-threading or asynchronous processing can help better utilize CPU/GPU resources.

5. **Hardware Acceleration**: Utilizing dedicated hardware accelerators like TensorRT on NVIDIA GPUs or Core ML on Apple devices can provide significant speedups.

### Difficulties Faced:

1. **Integration with Hardware**: Ensuring the code runs efficiently across different hardware platforms can be challenging, requiring thorough testing and optimization.

2. **Balancing Speed and Accuracy**: Finding the right trade-off between inference speed and model accuracy is a common difficulty.

3. **Model Deployment**: Deploying the model to production environments, especially with different backend requirements, can be complex.

4. **Real-time Processing**: Adapting the code to work in real-time applications, like live video feeds, may require significant refactoring and optimization.

5. **Environmental Variability**: Handling variations in lighting, background, and camera quality that affect model performance can be difficult.
