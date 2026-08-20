# Brain Tumor Segmentation System

A deep learning-based computer vision project for segmenting brain tumor
regions from multi-modal MRI scans. The system uses a U-Net-based
segmentation architecture with dynamic MRI modality reliability,
explainable AI, uncertainty estimation, radiomics-based tumor profiling,
and cross-modal corruption detection.

## Overview

Brain tumor segmentation is the task of identifying and outlining tumor
regions in MRI scans at the pixel level. This project works with the
BraTS 2020 dataset and uses four MRI modalities:

-   T1
-   T1ce
-   T2
-   FLAIR

The model learns from these complementary modalities and produces a
binary tumor segmentation mask.

## Key Features

-   Multi-modal MRI tumor segmentation
-   U-Net-based convolutional segmentation architecture
-   Dynamic modality reliability weighting for T1, T1ce, T2, and FLAIR
-   Dice loss for segmentation training
-   Grad-CAM-based model explainability
-   Monte Carlo Dropout for uncertainty-aware prediction
-   Radiomics-DL fusion for tumor profiling
-   Cross-modal hallucination/corruption detection
-   Graceful handling of a corrupted or missing MRI modality

## Dataset

The project uses the **BraTS 2020 Training and Validation Dataset**. MRI
volumes are provided in NIfTI format and include the four imaging
modalities along with segmentation masks.

The notebook downloads the dataset from Kaggle and processes the MRI
volumes for model training and evaluation.

## Tech Stack

-   Python
-   PyTorch
-   NumPy
-   NiBabel
-   MONAI
-   SciPy
-   Matplotlib
-   Torchvision
-   PyTorch Grad-CAM

## Model Architecture

The core segmentation model follows a U-Net-style encoder-decoder
architecture.

### Dynamic Modality Reliability

A custom modality reliability module learns the relative usefulness of
the four MRI modalities and dynamically weights them before
segmentation. This allows the model to give greater importance to
modalities that provide more useful information for the current input.

### Segmentation

The network uses convolutional blocks, pooling, an encoder-decoder
structure, and skip connections to preserve spatial information. The
model is trained using Dice loss, which is suitable for segmentation
problems where tumor pixels occupy a relatively small portion of the
image.

## Training

The model is trained using PyTorch and the Adam optimizer.

-   Training epochs: 50
-   Loss function: Dice Loss
-   Input modalities: 4
-   GPU acceleration: supported through CUDA/Colab

The training loss decreases substantially during training, indicating
that the model learns the segmentation task over the training process.

## Explainable AI

Grad-CAM is integrated to visualize the image regions contributing to
the segmentation prediction. This provides an interpretable heatmap that
can help understand where the model is focusing when identifying tumor
regions.

## Uncertainty Estimation

Monte Carlo Dropout is used to estimate pixel-level prediction
uncertainty.

The model performs multiple stochastic forward passes with dropout
enabled and calculates the mean prediction and standard deviation across
predictions. High-uncertainty regions can therefore be highlighted for
additional review.

## Tumor Profiling

The project also extracts handcrafted radiomic features from the
predicted tumor region, including:

-   Texture features
-   Shape features
-   Intensity statistics

These features are combined with the segmentation output to create a
tumor profiling component.

## Cross-Modal Corruption Detection

The system includes a mechanism for detecting corrupted or missing MRI
modalities. When a modality is corrupted, the system:

1.  Detects the affected modality.
2.  Computes reliability scores for the available modalities.
3.  Redistributes attention across modalities.
4.  Produces an alert identifying the affected channel.

This is intended to make the segmentation pipeline more robust when one
MRI modality is unavailable or unreliable.

## Evaluation

The segmentation model was evaluated using Dice Score, IoU, Precision,
Recall, and Accuracy.

  Metric          Score
  ------------ --------
  Dice Score     0.7577
  IoU            0.6099
  Precision      0.7771
  Recall         0.7391
  Accuracy       0.9985

These results represent the evaluation reported in the notebook and
should not be interpreted as a clinical validation study.

## Project Workflow

``` text
BraTS MRI Data
      |
      v
Load NIfTI Volumes
      |
      v
Preprocess MRI Modalities
      |
      v
T1 + T1ce + T2 + FLAIR
      |
      v
Dynamic Modality Reliability
      |
      v
U-Net Encoder-Decoder
      |
      v
Tumor Segmentation Mask
      |
      +--------------------+
      |                    |
      v                    v
   Grad-CAM          MC Dropout
      |                    |
      v                    v
Explainability        Uncertainty
      |
      +--------------------+
      |
      v
Radiomics + Tumor Profiling
```

## Installation

Install the required dependencies:

``` bash
pip install torch torchvision
pip install numpy nibabel monai scipy matplotlib
pip install grad-cam
```

## Running the Project

1.  Open the notebook in Google Colab or a CUDA-enabled Python
    environment.
2.  Install the required dependencies.
3.  Download and extract the BraTS 2020 dataset.
4.  Configure the dataset path.
5.  Run the preprocessing and data-loading cells.
6.  Train the segmentation model.
7.  Evaluate the segmentation output.
8.  Run the Grad-CAM, uncertainty estimation, radiomics, and modality
    corruption analysis sections.

## Expected Output

The project generates:

-   Predicted tumor segmentation masks
-   Segmentation evaluation metrics
-   Grad-CAM explanation heatmaps
-   Pixel-level uncertainty maps
-   Tumor radiomic profiles
-   Modality reliability/corruption alerts

## References

1.  Ronneberger et al. --- U-Net: Convolutional Networks for Biomedical
    Image Segmentation.
2.  Oktay et al. --- Attention U-Net: Learning Where to Look for the
    Pancreas.
3.  Isensee et al. --- nnU-Net: A Self-Configuring Method for Deep
    Learning-Based Biomedical Image Segmentation.
4.  Selvaraju et al. --- Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization.
5.  Menze et al. --- The Multimodal Brain Tumor Image Segmentation
    Benchmark (BraTS).
6.  Myronenko --- 3D MRI Brain Tumor Segmentation Using Autoencoder
    Regularization.
7.  Gal & Ghahramani --- Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning.
8.  Lambin et al. --- Radiomics: The bridge between medical imaging and
    personalized medicine.

## Disclaimer

This project is an educational/research implementation for brain tumor
image segmentation. It is not a clinical diagnostic system and should
not be used to make medical decisions.
