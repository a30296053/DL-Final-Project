# DL-Final-Project DRIAMS
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

CE6146 Introduction to Deep Learning Final Project - Group 1

## Project Overview
We utilize the **DRIAMS-A dataset**, specifically focusing on the resistance prediction of **_Staphylococcus epidermidis_** against the antibiotic **_Fusidic acid_**. By analyzing high-dimensional spectral data (6,000 intensity bins), this project aims to compare the efficacy of traditional classifiers versus deep neural networks in a clinical setting.

[Dataset Link (DRIAMS)](https://www.kaggle.com/datasets/drscarlat/driams/data)

## Repository Structure

```text
DL-Final-Project-Group1/
│
├── data/                        # Processed dataset storage
│   ├── fusidic_2017_XY(raw).npz # Combined raw dataset (Features + Labels)
│   ├── X_norm.npy               # Max-normalized feature matrix
│   ├── X_scaled.npy             # Standard-scaled feature matrix
│   └── Y.npy                    # Target labels (Resistant=1, Susceptible=0)
│
├── Preprocessing/               # Data extraction, cleaning & EDA
│   ├── 資料初步抓取/             # Raw metadata (Excel/CSV)
│   ├── visualize/               # Generated EDA plots (PCA, t-SNE, Spectra)
│   ├── build_dataset.py         # Script to merge raw text files into NPZ
│   ├── check&preprocess.py      # Script for normalization & saving .npy
│   └── vis.py                   # Script for exploratory data analysis
│
├── ML_Train/                    # Machine Learning Implementation (LR, RF, SVM)
│   ├── data/                    # (Local copy of data for ML)
│   ├── output/                  # Result figures (ROC, Confusion Matrix, Feature Imp.)
│   ├── data_preprocess.py       # ML-specific data splitting & scaling
│   ├── model.py                 # Model definitions & evaluation logic
│   └── main.py                  # Entry point to run the ML pipeline
│
├── DL_Train/                    # Deep Learning Implementation (PyTorch)
│   ├── Config.py                # Hyperparameters (Epochs, Input Size)
│   ├── DataSets.py              # Custom PyTorch Dataset class
│   ├── MODEL.py                 # Funnel MLP Model Architecture
│   └── main.py                  # Entry point for training & testing DL models
│
├── Report.pdf                   # Project Report
├── README.md                    # Project documentation
└── requirement.txt              # List of Python dependencies
```

## Installation

### 1. Clone the Repository
First, clone the repository to your local machine:
```bash
git clone [https://github.com/your-username/DL-Final-Project-Group1.git](https://github.com/your-username/DL-Final-Project-Group1.git)
cd DL-Final-Project-Group1
```

### 2. Install Dependencies
It is recommended to use a virtual environment (optional but good practice). Install all required libraries using requirements.txt:

```Bash
pip install -r requirements.txt
```

### 3. Data Setup
Since the dataset is large, it is not included in this repository.

So if you want to do data preprocessing by yourself, Please download the DRIAMS-A dataset from Kaggle.

Create a folder named data in the Preprocessing/ directory.

Place the binned_6000 folder and id folder (containing CSVs) into Preprocessing/data/.

Note: Ensure the file paths match the structure in Directory Structure

## Usage

Follow these steps to reproduce the results.

### Step 1: Data Preprocessing
Navigate to the `Preprocessing` directory to construct and normalize the dataset.

1. **Build Dataset**: Convert raw text files into a unified matrix.
   ```bash
   cd Preprocessing
   python build_dataset.py
   ```
2. Normalize & Scale: Generate X_norm.npy and Y.npy.
    ```Bash
    python check&preprocess.py
    ```
3. (Optional) Visualization: Run EDA to view spectra plots.
    ```Bash
    python vis.py
    ```

### Step 2: Prepare Training Data
Crucial Step: Before training, you must ensure the data files (X_norm.npy, X_scaled.npy, Y.npy) from the Preprocessing/ folder are in the data directories of the training modules:

Machine Learning to: ML_Train/data/

Deep Learning to: DL_Train/data/

### Step 3: Machine Learning (ML)
Train and evaluate traditional classifiers (Logistic Regression, Random Forest, SVM).
```Bash
cd ../ML_Train
python main.py
```
Output:
- Console: Accuracy, F1-Score, and Classification Reports.
- Figures: Saved in ML_Train/output/ (ROC Curves, Confusion Matrices).

### Step 4: Deep Learning (DL)
Train the Funnel MLP model using PyTorch.

```Bash
cd ../DL_Train
python main.py
```
Interactive Mode:
- Type y to Train a new model (Logs F1-score & saves best model).
- Type n to Test the best saved model on the test set.

## Results

Detailed performance metrics, confusion matrices, and model comparisons are documented in the full project report.

**[View Full Project Report (PDF)](https://drive.google.com/file/d/1HG-wg0946IoFCqldsdPSvHjXsnuFaTC7/view?usp=sharing)**

We evaluated four models based on **Recall** and **F1-Score**. The Deep Learning model demonstrated superior sensitivity in detecting resistant cases.

| Model | Recall (Resistant) | F1-Score |
| :--- | :---: | :---: |
| **Deep Learning (MLP)** | **0.80** | **0.76** |
| SVM (RBF) | 0.76 | 0.75 |
| Random Forest | 0.72 | 0.70 |

Some of the files do not post on the github due to the size limitation.