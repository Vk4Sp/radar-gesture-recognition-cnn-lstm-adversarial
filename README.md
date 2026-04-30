# Radar-Based Gesture Recognition using CNN-LSTM with Doppler Spread and Adversarial Training

## Overview
This project implements radar-based hand gesture recognition using range-Doppler data from the Soli dataset.

The model combines:
- CNN for spatial feature extraction
- LSTM for temporal modeling
- Doppler spread feature for motion complexity
- FGSM adversarial training for robustness

## Method
- Baseline: CNN + LSTM
- Proposed: CNN + LSTM + Doppler Spread
- Final: CNN + LSTM + Spread + FGSM

FGSM perturbation:
X_adv = X + epsilon * sign(gradient)
epsilon = 0.01

## Results
| Model | Accuracy |
|------|----------|
| CNN + LSTM | 86.36% |
| CNN + LSTM + Spread | 85.27% |
| CNN + LSTM + Spread + FGSM | 87.00% |

## How to Run
1. Update dataset path inside the code
2. Run:
   python train_spread.py

## Dataset
This project uses the Soli radar dataset for gesture recognition.

The dataset is not included in this repository. It can be accessed from:
https://github.com/google/soli

Please download the dataset and update the dataset path in the code before running.

## Files
- data_loader.py
- model_Spread.py
- train_spread.py
- visualize.py

## Notes
- No pretrained models used
- Focus on robustness and interpretability