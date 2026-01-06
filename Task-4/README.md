\# Task Phase 4 – Convolutional Neural Networks



This folder contains the work completed for \*\*Task Phase 4\*\*.  

It includes two tasks implemented using \*\*PyTorch\*\*:



\- \*\*Task 1:\*\* Basic CNN implementation from scratch on Fashion-MNIST  

\- \*\*Task 2:\*\* Transfer Learning and Fine-Tuning on DeepWeeds and FER2013 datasets  



All models were evaluated using standard classification metrics including accuracy, precision, recall, F1-score, and confusion matrices.



---



\## Task 1: Basic CNN Implementation (Fashion-MNIST)



\### Objective

To design and train a custom Convolutional Neural Network (CNN) from scratch for image classification on the Fashion-MNIST dataset, without using any pre-trained models.



---



\### Dataset

\- \*\*Fashion-MNIST\*\*

\- 70,000 grayscale images (28 × 28)

\- 10 clothing categories:

&nbsp; - T-shirt/top, Trouser, Pullover, Dress, Coat  

&nbsp; - Sandal, Shirt, Sneaker, Bag, Ankle boot



---



\### Model Architecture

\- Custom CNN designed manually

\- Minimum of \*\*3 convolutional layers\*\*

\- ReLU activations

\- Max-pooling layers for spatial downsampling

\- Fully connected classifier head

\- Dropout for regularization



---



\### Data Preprocessing and Augmentation

\- Pixel normalization

\- Random horizontal flipping

\- Random rotations

\- Train / Validation / Test split used



---



\### Training Setup

\- Optimizer: Adam

\- Loss Function: Cross-Entropy Loss

\- Epochs: 20

\- Validation-based performance monitoring



---



\### Results (Fashion-MNIST)



\#### Training and Validation Accuracy

\- Training accuracy steadily increased across epochs

\- Validation accuracy peaked at \*\*92.87%\*\*



\#### Final Test Performance



| Metric | Value |

|------|------|

| Accuracy | \*\*92.0%\*\* |

| Macro F1-score | \*\*0.92\*\* |

| Weighted F1-score | \*\*0.92\*\* |



\#### Class-wise Observations

\- Very high performance on classes like \*\*Trouser\*\*, \*\*Bag\*\*, and \*\*Ankle boot\*\*

\- Slight confusion between visually similar classes such as \*\*Shirt\*\* and \*\*T-shirt/top\*\*

\- Demonstrates effective hierarchical feature learning using a CNN trained from scratch



---



\## Task 2: Transfer Learning and Fine-Tuning



\### Objective

To apply transfer learning and fine-tuning strategies using pre-trained CNN architectures on two real-world datasets and compare different adaptation approaches under limited computational resources.



---



\## Dataset 1: DeepWeeds



\### Dataset Description

\- ~17,500 RGB images

\- 9 weed species classes

\- Natural outdoor environments with varying lighting and backgrounds



---



\### Model Architecture

\- \*\*ResNet18 (ImageNet pre-trained)\*\*

\- Entire convolutional backbone frozen

\- Final fully connected layer replaced and trained



---



\### Rationale

\- ImageNet features capture general visual patterns (edges, textures, shapes)

\- These features transfer effectively to outdoor weed imagery

\- Freezing the backbone reduces overfitting and training time



---



\### Data Augmentation

\- Random resized cropping

\- Horizontal flipping

\- Random rotation

\- Color jitter

\- ImageNet normalization



---



\### Results (DeepWeeds)



| Metric | Value |

|------|------|

| Accuracy | \*\*78.0%\*\* |

| Weighted F1-score | \*\*0.78\*\* |

| Macro F1-score | \*\*0.71\*\* |



\*\*Key Findings\*\*

\- Strong recall for dominant classes such as \*Negative\*

\- Moderate confusion among visually similar weed species

\- Effective transfer of ImageNet features to agricultural imagery



---



\## Dataset 2: FER2013 (Facial Expression Recognition)



\### Dataset Description

\- Grayscale facial images

\- 7 emotion classes

\- Low resolution and strong class imbalance



---



\### Model Architecture

\- \*\*ResNet18 (trained without ImageNet weights)\*\*

\- Entire network frozen except final convolutional block (layer4)

\- Classifier layer trained from scratch



---



\### Rationale

\- Facial expression data differs significantly from ImageNet object images

\- Partial fine-tuning allows adaptation of high-level features

\- Full fine-tuning was infeasible under CPU-only constraints



---



\### Data Augmentation

\- Horizontal flipping

\- Random rotation

\- Image resizing and normalization



---



\### Results (FER2013)



| Metric | Value |

|------|------|

| Accuracy | \*\*44.0%\*\* |

| Weighted F1-score | \*\*0.41\*\* |

| Macro F1-score | \*\*0.34\*\* |



\*\*Key Findings\*\*

\- High recall for the \*Happy\* class due to dataset imbalance

\- Lower performance on minority classes such as \*Disgust\*

\- Results consistent with known challenges of FER2013



---



\## Comparison of Tasks



| Aspect | Task 1 (Fashion-MNIST) | Task 2 (Transfer Learning) |

|-----|----------------------|---------------------------|

| Model Type | CNN from scratch | Pre-trained ResNet18 |

| Dataset Complexity | Low | High |

| Training Cost | Moderate | Low–Moderate |

| Transfer Learning | No | Yes |

| Fine-Tuning Strategy | N/A | Feature extraction vs partial fine-tuning |



---



\## Overall Conclusions



\- CNNs trained from scratch perform very well on simpler datasets like Fashion-MNIST.

\- Transfer learning significantly reduces training cost for complex real-world datasets.

\- Fine-tuning strategies must be chosen based on dataset characteristics and hardware constraints.

\- Class imbalance remains a major challenge, especially for facial expression recognition.



---



\## Tools and Frameworks

\- PyTorch

\- Torchvision

\- NumPy, Pandas

\- Matplotlib, Seaborn

\- Scikit-learn



---



\## Author

\*\*Vaibhav Chourasia\*\*



