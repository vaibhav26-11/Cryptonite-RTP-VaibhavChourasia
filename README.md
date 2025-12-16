

---



\## Task 1: MLP From Scratch (Wine Quality Dataset)



\### Objective

Predict \*\*wine quality scores\*\* using a \*\*fully manual Multilayer Perceptron\*\*, without using any deep learning framework.



\### Dataset

\- Wine Quality Dataset (Red + White)

\- Total samples: \*\*6497\*\*

\- Features: \*\*12\*\*

\- Target: Wine quality score (regression)



\### Implementation Details

\- Libraries used: \*\*NumPy, Pandas, scikit-learn, Matplotlib\*\*

\- Manual implementation of:

&nbsp; - Forward propagation

&nbsp; - ReLU activation

&nbsp; - Backpropagation

&nbsp; - Gradient Descent

&nbsp; - He weight initialization

&nbsp; - L2 regularization

&nbsp; - Early stopping

\- Train / Validation / Test split: \*\*70% / 15% / 15%\*\*



\### Final Performance

| Dataset | MSE | RMSE | MAE | R² |

|-------|-----|------|-----|----|

| Train | 0.3906 | 0.6250 | 0.4876 | 0.4940 |

| Val   | 0.4497 | 0.6706 | 0.5212 | 0.3826 |

| Test  | 0.4655 | 0.6823 | 0.5379 | 0.3808 |



\### Analysis

\- The model learns meaningful nonlinear relationships.

\- Moderate R² is expected due to the noisy nature of wine quality labels.

\- Early stopping successfully prevents overfitting.

\- Residual and prediction plots are included for error analysis.



---



\## Task 2: MLP Using Framework (Adult Income Dataset)



\### Objective

Classify whether a person’s annual income is \*\*greater than \\$50K\*\*.



\### Dataset

\- Adult Income Dataset

\- Total samples after cleaning: \*\*45,222\*\*

\- Input features after encoding: \*\*104\*\*



\### Implementation Details

\- Framework used: \*\*PyTorch\*\*

\- Architecture:

&nbsp; - Fully connected deep MLP

&nbsp; - ReLU activations

&nbsp; - Dropout regularization

&nbsp; - Sigmoid output layer

\- Loss: Binary Cross-Entropy

\- Train / Validation / Test split: \*\*70% / 15% / 15%\*\*



\### Final Test Performance

\- \*\*Accuracy:\*\* 84.82%



\#### Classification Report

\- Strong performance on majority class

\- Reasonable recall on minority (>50K) class

\- Confusion matrix and loss curves included for evaluation



\### Analysis

\- Deep architecture captures complex categorical interactions.

\- Dropout reduces overfitting.

\- Adam optimizer ensures stable and fast convergence.



---



\## Task 3: Theory Reports



Detailed LaTeX reports covering theory and mathematics behind MLPs:



1\. \*\*Backpropagation and Gradient-Based Optimization\*\*

&nbsp;  - Forward \& backward propagation

&nbsp;  - Chain rule derivations

&nbsp;  - Gradient descent, Momentum, RMSprop, Adam

&nbsp;  - Vanishing and exploding gradients

&nbsp;  - Learning rate strategies



2\. \*\*Regularization in MLPs\*\*

&nbsp;  - Overfitting and underfitting

&nbsp;  - L1, L2 regularization and weight decay

&nbsp;  - Dropout

&nbsp;  - Batch Normalization \& Layer Normalization

&nbsp;  - Early stopping



3\. \*\*Weight Initialization and Training Stability\*\*

&nbsp;  - Xavier and He initialization (with derivations)

&nbsp;  - Choice of activation functions

&nbsp;  - ReLU and dying ReLU problem

&nbsp;  - Gradient clipping

&nbsp;  - Softmax and numerical stability





