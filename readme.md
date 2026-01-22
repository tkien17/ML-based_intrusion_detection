# Machine Learning-based Intrusion Detection System (UNSW-NB15)

This repository contains a replication of experimental results for a Network Intrusion Detection System (NIDS) using the **UNSW-NB15 dataset**.

The project implements and compares two feature reduction techniques—**Feature Selection (Correlation-based)** and **Feature Extraction (PCA)**—across five different machine learning models for both Binary and Multi-class classification.

---

## 📂 Project Structure

```
Data/                    # UNSW-NB15 dataset (Training and Testing sets)
replication.py           # Main Python script with data processing and evaluation
train.ipynb              # Jupyter Notebook with step-by-step execution and visualizations
requirements.txt         # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. Install required libraries using:

```bash
pip install -r requirements.txt
```

### Running the Project

**Option 1: Run the Python script**
```bash
python replication.py
```

**Option 2: Open the Jupyter Notebook**
Open `train.ipynb` in Jupyter Notebook or VS Code to view step-by-step execution and visualized results.

---

## 🧠 Methodology

The project follows these key implementation stages:

### 1. Data Preprocessing

- Dropping unnecessary columns (e.g., `id`)
- Handling missing values in the `service` feature by replacing `'-'` with `'other'`
- Applying One-Hot Encoding for categorical features: `proto`, `service`, and `state`
- Aligning testing set columns to match training set after encoding

### 2. Feature Reduction (Target K=4)

**Feature Selection (Correlation-based)**
- Utilizes a Correlation Matrix to identify top K features
- Selects features with highest average correlation

**Feature Extraction (PCA)**
- Employs Principal Component Analysis for dimensionality reduction
- Applies Min-Max scaling before PCA
- Reduces to K principal components

### 3. Model Training and Evaluation

Five algorithms evaluated for both **Binary** (Normal vs. Attack) and **Multi-class** (specific attack categories) classification:

1. Decision Tree
2. Random Forest
3. K-Nearest Neighbors (KNN)
4. Multi-Layer Perceptron (MLP)
5. Naive Bayes

---

## 📊 Key Findings

### Feature Extraction (PCA)

| Aspect | Details |
|--------|---------|
| **Best For** | Complex tasks like Multi-class classification with small K (e.g., K=4) |
| **Best Model** | Multi-Layer Perceptron (MLP) |
| **Key Strength** | Captures global data patterns, leading to higher accuracy in detecting diverse attack types |

### Feature Selection (Correlation)

| Aspect | Details |
|--------|---------|
| **Best For** | Real-time systems requiring ultra-low latency with larger K (e.g., K=8 or 16) |
| **Best Model** | Decision Tree |
| **Key Strength** | Superior inference speed and lower training time, ideal for high-speed network environments |

### Summary Comparison

| Metric | Feature Extraction | Feature Selection |
|--------|-------------------|-------------------|
| Accuracy (at K=4) | Higher | Lower |
| Inference Time | Higher | Lower |
| Training Time | Higher | Lower |
| Best Classifier | MLP | Decision Tree |

---

## 📝 Conclusion

The experimental results demonstrate important trade-offs between the two approaches:
- **Use Feature Extraction (PCA)** when accuracy is prioritized in complex classification tasks
- **Use Feature Selection** when speed and real-time performance are critical requirements
