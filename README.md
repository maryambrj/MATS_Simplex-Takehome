# Mechanistic Interpretability Analysis

This project explores the training and internal mechanics of a transformer model trained on synthetic data. It is divided into two main parts: training and evaluation (`part1`) and geometric analysis of internal representations (`part3`).

## Project Structure

- **`part1/`**: Contains the model architecture, training scripts, and dataset generation.
  - `src/model.py`: Decoder-only transformer implementation.
  - `train.py`: Training pipeline for the synthetic task.
  - `run_part1.py`: End-to-end script for data generation, training, and initial evaluation.
- **`part3/`**: Contains scripts for mechanistic interpretability and geometric analysis.
  - `extract_activations.py`: Extracts residual stream activations from the trained model.
  - `analyze_geometry.py`: Performs PCA and linear probe analysis on the extracted activations.
  - `results/`: Contains visualization plots and summary metrics.

## Setup

Install the required dependencies:

```bash
pip install torch numpy pandas matplotlib scikit-learn
```

## Usage

### Part 1: Training and Basic Evaluation

To generate data, train the model, and see basic sequence examples and loss plots, run:

```bash
cd part1
python run_part1.py
```

### Part 3: Geometric Analysis

After training the model in Part 1, you can perform the geometric analysis:

1. **Prepare Analysis Dataset**:
   ```bash
   cd part3
   python analysis_dataset.py
   ```

2. **Extract Activations**:
   ```bash
   python extract_activations.py
   ```

3. **Run Geometric Analysis**:
   ```bash
   python analyze_geometry.py
   ```
   This will generate plots and a `summary_metrics.txt` in the `part3/results/` directory.
