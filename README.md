# Paddy Disease Classifier using PyTorch

An end-to-end Deep Learning project that classifies rice leaf diseases using a Convolutional Neural Network (CNN) built with PyTorch.

The project includes:

- Custom CNN architecture
- Training pipeline
- Streamlit Web Application
- Model checkpointing
- Early Stopping
- Batch Normalization
- Data Augmentation
- Real-world testing

---
# Lessons Learned

This project taught me several practical machine learning lessons:

- High validation accuracy does not guarantee real-world performance.
- Dataset quality is often more important than model complexity.
- Proper evaluation requires testing on images completely outside the training distribution.
- Batch Normalization and Data Augmentation improved convergence, but could not fully solve the generalization problem.
- Building AI for agriculture requires diverse, expert-labeled datasets rather than relying solely on benchmark datasets.

# Motivation

Rice is one of the most important crops in Bangladesh.

Many farmers cannot easily identify diseases in the early stage.

The goal of this project is to build an AI model that can assist farmers by identifying rice diseases from a simple smartphone image.

---
# Dataset

Training Dataset:

**Paddy Doctor Dataset**

Number of Classes:

10

Classes:

- bacterial_leaf_blight
- bacterial_leaf_streak
- bacterial_panicle_blight
- blast
- brown_spot
- dead_heart
- downy_mildew
- hispa
- normal
- tungro

---

# Model Architecture

Custom CNN built from scratch.

Architecture:

Input Image (128×128)

↓

Conv2D (32)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Conv2D (64)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Conv2D (128)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Conv2D (256)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Conv2D (512)

↓

BatchNorm

↓

ReLU

↓

MaxPool

↓

Flatten

↓

Linear (128)

↓

Dropout

↓

Output Layer (10 Classes)

---

# Data Augmentation

Training images are augmented using:

- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Resize
- RandomAffine
- RandomErasing
- Tensor Conversion

---

# Training Features

✔ Batch Normalization

✔ Dropout

✔ Adam Optimizer

✔ Learning Rate Scheduler

✔ Early Stopping

✔ Model Checkpoint Saving

---

# Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

---

# Project Structure

```
Paddy-Disease-Classifier/

│

├── main.py

├── model.py

├── best_model.pth

├── label_encoder.pkl

├── requirements.txt

├── README.md

│

├── assets/

│      demo.gif

│      screenshot.png

│

└── training/
       train.ipynb
```

---

# Results

Validation Accuracy:

**96%+**

Test Accuracy:

**93%+**

Although the validation accuracy is high, the model still struggles with many real-world images collected from Google and agricultural sources.

---

# What I Learned

One of the biggest lessons from this project was:

> High Validation Accuracy does **NOT** always mean good real-world performance.

During testing, I found that images outside the training dataset often produced incorrect predictions.

Possible reasons include:

- Domain Shift
- Different lighting conditions
- Different cameras
- Different disease stages
- Background variations
- Dataset limitations

This project helped me understand the importance of **generalization**, not just benchmark accuracy.

---

# Current Limitations

The current model:

- Uses a custom CNN trained from scratch.
- Generalizes poorly on many unseen images.
- Depends heavily on the quality of the training dataset.
- May become overconfident on unfamiliar samples.

---

# Future Improvements

I plan to improve this project by:

- Fine-tuning EfficientNet-B0
- Using Transfer Learning
- Collecting expert-labeled datasets
- Adding confidence thresholding
- Testing on thousands of real-world images
- Improving dataset diversity
- Deploying a mobile-friendly version

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Paddy-Disease-Classifier.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run main.py
```

---

# Future Goal

My long-term goal is to build an AI-powered assistant that can help farmers identify rice diseases using only a smartphone.

Rather than focusing solely on achieving high benchmark accuracy, I aim to develop a model that performs reliably under real-world conditions.

---

# Acknowledgements

- PyTorch
- Streamlit
- Kaggle
- Paddy Doctor Dataset
- Bangladesh Agricultural Research Community

---

# Connect With Me

LinkedIn:

https://linkedin.com/in/mohiuddin-mahady

GitHub:

https://github.com/mahady13

Email:

mohiuddinmahady@gmail.com