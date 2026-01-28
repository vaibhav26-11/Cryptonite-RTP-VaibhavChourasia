# Task-5: Weather Forecasting and Character-Level Language Modeling

This repository contains the implementation and results for **Task-5 (Sequence Models)**, which includes:

1. **Time-series weather forecasting for Manipal** using RNN, GRU, and LSTM models  
2. **A character-level LSTM language model** trained on *The Office (US)* dialogue dataset  

All experiments were implemented in **PyTorch**, following proper preprocessing, chronological data splitting, and evaluation protocols.

---

## Setup

### Environment
- Python 3.8+
- PyTorch
- NumPy, Pandas
- Matplotlib
- scikit-learn

### Repository Structure
- `Weather_Forecasting.ipynb` – Time-series forecasting using RNN, GRU, and LSTM  
- `The_Office_Generator.ipynb` – Character-level LSTM language model  
- `Climate_Analysis_EDA.ipynb` – Climatic analysis and trend study for Manipal  
- `reports/` – Short technical reports (LaTeX)  
- `README.md` – Project overview and results  

---

## 1. Manipal Weather Forecasting

### Task Description
Next-day prediction of:
- **Mean temperature at 2m (°C)**
- **Daily precipitation sum (mm)**

using historical atmospheric sensor readings.

The problem was formulated as a **many-to-one time-series regression task**, where a fixed window of past days is used to predict the next day’s value.

---

### Data & Preprocessing
- Dataset: Manipal daily weather data (2011–2026)
- Chronological split:
  - **Training:** before 1 Jan 2023  
  - **Validation:** 1 Jan 2023 – 3 Jan 2025  
  - **Test:** 4 Jan 2025 – 4 Jan 2026  
- Missing values handled using forward fill
- Multivariate inputs scaled using **StandardScaler**, fit on the training split only

---

### Models Evaluated
- Vanilla **RNN**
- **GRU**
- **LSTM**

All models were trained using:
- Sliding window length: **30 days**
- Optimizer: Adam
- Loss function: Mean Squared Error (MSE)

---

### Model Comparison (Temperature Prediction)

| Model | RMSE | MAE |
|------|------|-----|
| RNN | 0.3342 | 0.2567 |
| GRU | 0.3430 | 0.2615 |
| LSTM | 0.3408 | 0.2606 |

**Observation:**  
All three architectures perform comparably for short-horizon forecasting.  
In this configuration, the simpler RNN marginally outperforms GRU and LSTM, likely due to the smooth and highly autocorrelated nature of daily temperature data and the short prediction horizon.

---

### Univariate vs Multivariate LSTM

| Setup | RMSE | MAE |
|------|------|-----|
| Univariate LSTM (Temperature only) | 0.3487 | 0.2640 |
| Multivariate LSTM | 0.3408 | 0.2606 |

**Conclusion:**  
Including additional atmospheric features leads to measurable improvement over the univariate baseline.

---

### Precipitation Forecasting (Multivariate LSTM)

- Precipitation is significantly noisier and more difficult to predict
- The model captures major rainfall events but exhibits higher variance

Loss curves and prediction plots are included in the notebook.

---

### Visualizations Included
- Training vs validation loss curves (best-performing model)
- Ground truth vs predicted temperature (test period: 2025–2026)
- Ground truth vs predicted precipitation (test period: 2025–2026)

---

## 2. Character-Level LSTM: The Office (US)

### Task Description
Train a **character-level language model** to generate dialogue-like text from *The Office (US)* script.

---

### Data & Tokenization
- Character-level vocabulary (72 unique characters)
- Full script encoded as character IDs
- Context length: **100 characters**
- Data split: **90% training / 10% testing**

---

### Model Architecture
- Character embedding layer
- **2-layer LSTM** with dropout
- Fully connected output layer over vocabulary
- Cross-entropy loss (implicit softmax)

---

### Evaluation Metric
- **Perplexity**

**Final Test Perplexity:** **3.4030**  
(This falls within the expected range for character-level language models.)

---

### Text Generation & Temperature Sampling

A custom generation function was implemented:

```python
generate_script(seed_text, temperature, num_chars)
```
#### Best Generation Example

Seed text: "Michael: Okay everyone, listen up. "

Temperature used: 0.7

Test Perplexity: 3.4030

Sample Generated Output (excerpt):

Michael: Okay everyone, listen up. The boss for the boss, it's for the most meatball.
Darryl: That's very good.
Michael: Well, I would like to take my eyes or...


#### Temperature Analysis

Temperature = 0.3: Generates highly predictable and repetitive text with strong grammatical structure.

Temperature = 0.7: Produces the best balance between coherence and creativity, generating dialogue that resembles the original script style.

Temperature = 1.0: Produces more diverse and creative outputs, but with occasional grammatical inconsistencies.


## Conclusions

Multivariate sequence models improve weather forecasting accuracy over univariate baselines.

RNN, GRU, and LSTM perform similarly for short-term temperature prediction, with only minor differences in error metrics.

Proper windowing, feature scaling, and chronological data splits are critical for stable time-series forecasting.

The character-level LSTM successfully learns statistical patterns in dialogue text from The Office (US) dataset.

Perplexity values fall within the expected range for character-level language modeling.

Temperature-based sampling provides effective control over the trade-off between coherence and creativity in generated text.
