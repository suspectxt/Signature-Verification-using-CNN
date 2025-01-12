# Image Classification with Convolutional Neural Networks (CNN)

## Project Overview

This project demonstrates the use of a Convolutional Neural Network (CNN) to classify images into two categories: real and forged signatures. The dataset used contains images of real private signatures, which cannot be shared publicly due to privacy concerns. The goal of the project is to train a CNN model to accurately classify the images and evaluate the model using accuracy and confusion matrix metrics.

## Dataset

- The dataset consists of signature images categorized into two classes: `real` and `forged`.
- **Due to privacy concerns, the dataset cannot be shared publicly.** However, you can prepare a similar dataset by collecting signature images, labeling them accordingly, and preprocessing them in the same way.

### How to Use the Dataset Locally

1. **Prepare the Data**:
   - Create two folders: `train` and `test`.
   - Inside each folder, create subfolders for each category: `real` and `forged`.
   - Place the signature images into the respective subfolders. Make sure to resize the images to 300x450 pixels and convert them to grayscale.

2. **Preprocessing**:
   - The preprocessing steps are included in the code and will handle converting the images to grayscale, resizing, and applying thresholding.
   - You can use the `preprocess.py` script to preprocess the data, which can then be used for training and testing the model.

## Data Augmentation

To improve the model's generalization, **data augmentation** is applied to both the training and testing datasets. The following augmentation techniques are used:

1. **Noise**: Random noise is added to the images to simulate real-world distortions.
2. **Blur**: Gaussian blur is applied to mimic the effect of blurry signatures.
3. **Rotation**: The images are rotated randomly by up to 27 degrees to simulate variations in writing angles.

### Augmentation Process:
For each image in both the `train` and `test` sets (both real and forged images), 25 augmented versions are generated using the following steps:

- **Noise**: Random noise is added to the image, with values ranging between -30 and +30 for each pixel.
- **Blur**: Gaussian blur is applied with a kernel size of 5.
- **Rotation**: Each image is randomly rotated between -27 and +27 degrees.

The augmented images are saved with a suffix (`_transformed`) appended to the original filenames.

### How to Apply Data Augmentation

To apply data augmentation to your dataset, use the `imggen` function, which will apply augmentations to both the `train` and `test` directories for real and forged signatures.

Example:
```python
imggen('person1')  # Augments both train and test datasets for person1
```

This will generate 25 augmented versions of each image in both the training and testing datasets.

## Model Architecture

The CNN model architecture consists of:
- Two convolutional layers with 64 filters each and ReLU activation.
- Max pooling layers after each convolutional layer.
- A fully connected dense layer with 1 output unit using the sigmoid activation function for binary classification.

### Model Summary:
- Input Shape: (300, 450, 1)
- Layers: 
  1. Conv2D (64 filters, 3x3 kernel)
  2. MaxPooling2D (3x3 pool)
  3. Conv2D (64 filters, 3x3 kernel)
  4. MaxPooling2D (3x3 pool)
  5. Flatten
  6. Dense (1 unit, sigmoid activation)

## Results

- **Accuracy**: 89.45%
- **Confusion Matrix**:
  ```
  [[126  21]
   [  0  52]]
  ```
  The model achieves high accuracy, with very few false positives and no false negatives.

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/suspectxt/Stock-Prediction-Using-Bidirectional-LSTM.git
   ```

2. Prepare the dataset locally (see instructions above).

3. Apply data augmentation to the dataset (optional but recommended):
   ```python
   imggen('person1')  # Augments both train and test datasets for person1
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/signature_verification.ipynb
   ```

## Dependencies

- Python 3.x
- TensorFlow/Keras
- OpenCV
- scikit-learn
- matplotlib
- numpy
- pandas

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/suspectxt/Stock-Prediction-Using-Bidirectional-LSTM/blob/main/LICENSE) file for details.
