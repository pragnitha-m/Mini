# Glaucoma Detection Using Machine Learning

## Overview

This project aims to develop an automated glaucoma detection system using machine learning techniques. It employs Convolutional Neural Networks (CNNs) and Graph Neural Networks (GNNs) to analyze retinal images for identifying glaucoma.

## Objectives

- Detect glaucoma from retinal images accurately.
- Provide a reproducible framework for medical image analysis.

## Features

- **Data Preprocessing**: Includes resizing, normalization, and augmentation of retinal images to prepare for model training.
- **CNN Implementation**: Utilizes ConvNeXt architecture for feature extraction and classification.
- **GNN Exploration**: Incorporates graph-based learning for enhanced pattern recognition.
- **Model Evaluation**: Offers performance metrics like accuracy, precision, recall, and F1-score.
- **Visualization**: Includes training and validation accuracy/loss plots.

## Requirements

To replicate this project, install the following dependencies:

```bash
pip install tensorflow
pip install numpy
pip install matplotlib
pip install scikit-learn
```

## Usage

1. **Data Preparation**:

   - Place the retinal images in the appropriate directory structure as expected by the notebook.
   - Update the `dataPath` variable in the code to point to your dataset location.

2. **Model Training**:

   - Run the notebook to preprocess data, build the CNN model, and train it on your dataset.

3. **Evaluation**:

   - Evaluate the trained model on the test dataset to obtain metrics like accuracy and classification reports.

4. **Visualization**:

   - Use the included plotting functions to visualize training progress and performance.

## Directory Structure

Ensure the following structure for the dataset:

```
project_directory/
  |
  |-- data/
      |-- train/
      |-- validation/
      |-- test/
```

## Results

- **Model Performance**: The trained model achieved an accuracy of **97%** and a precision of **97.5%** on the test dataset.
- **Visualization**: Plots illustrating the training/validation accuracy and loss curves.

## Future Work

- Deploy the model as a web application for real-world usability.

## Acknowledgments

- **Team Members**: G.Sree Varshini, Sneha Reddy.M, M.Pragnitha
- **Mentor**: Dr. Mukhtar Ahmad Sofi, Associate Professor
- Libraries used: TensorFlow, NumPy, Matplotlib, Scikit-learn
- Contributions from team members and mentors.

## License

This project is licensed under the MIT License.

