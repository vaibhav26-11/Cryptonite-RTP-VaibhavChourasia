# Task-5: Weather Forecasting and Character-Level Language Modeling

This repository contains the implementation and results for **Task-5**, which includes:
1. Weather forecasting for Manipal using RNN-based models
2. A character-level LSTM language model trained on *The Office (US)* dialogues

All experiments were implemented in PyTorch and evaluated using appropriate metrics.

---

## Setup

### Environment
- Python 3.8+
- PyTorch
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

### Files
- `Weather_Forecasting.ipynb` – Manipal weather prediction using RNN, GRU, and LSTM
- `The_Office_Generator.ipynb` – Character-level LSTM language model
- `Climate_Analysis_EDA.ipynb` – Climatic analysis and trend study

---

## Results

### 1. Manipal Weather Forecasting

**Task:**  
Next-day prediction of:
- Mean temperature at 2m (°C)
- Daily precipitation sum (mm)

**Best Model:** Multivariate LSTM  
**Evaluation Metrics:** MAE, RMSE

| Variable | MAE | RMSE |
|--------|------|------|
| Temperature (°C) | 0.2646 | 0.3430 |
| Precipitation (mm) | 0.4549 | 0.9925 |

**Observations:**
- Multivariate models outperform univariate baselines
- LSTM performs better than GRU and vanilla RNN
- Temperature predictions closely follow seasonal trends
- Precipitation predictions capture major rainfall events but remain noisier due to high variability

---

### 2. Character-Level LSTM: The Office (US)

**Task:**  
Next-character prediction and dialogue generation from script text.

**Model:**
- Character-level vocabulary and tokenization
- Embedding → LSTM stack → Dense + Softmax
- Cross-entropy loss

**Evaluation Metric:** Perplexity

**Final Test Perplexity:** **3.4030**

**Generation Results:**
- Generated text follows the format `[Person]: [Dialogue]`
- Valid English words and dialogue structure emerge from character-level learning
- Temperature-based sampling shows clear differences:
  - Low temperature (0.3): more repetitive and conservative text
  - Medium temperature (0.7): balanced coherence and creativity
  - High temperature (1.0): increased creativity with noisier output

A custom generation function was implemented:
```python
best_seed = "Michael: Okay everyone, listen up. "
best_temperature = 0.7
best_sample = generate_script(
    best_seed,
    temperature=best_temperature,
    num_chars=500
)
```


---

## Plots

### Weather Forecasting
- Ground truth vs prediction (Temperature)
- Ground truth vs prediction (Precipitation)
- Training vs validation loss curves for RNN, GRU, and LSTM models

### Language Model
- Training and validation loss curves
- Perplexity curve
- Sample generated dialogue outputs at different temperature values (0.3, 0.7, 1.0)

---

## Conclusions

- Multivariate LSTM models provide the best performance for Manipal weather forecasting.
- Temperature predictions achieve low error and closely follow seasonal patterns, while precipitation remains harder to predict due to high variability.
- Proper windowing, scaling, and feature selection are crucial for stable time-series forecasting.
- The character-level LSTM successfully learns statistical patterns in dialogue text and generates structured script-like output.
- Temperature-based sampling effectively controls the trade-off between coherence and creativity in generated text.


