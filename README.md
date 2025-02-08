# Gaussian Reservoir State Analysis (GRSA)

## Paper
Under review

## Architecture
<img src="figures/schematic_RC_vs_GRSA.svg" width="800">

## Demonstration of Online Time Series Classification (Regression vs. GRSA)

|![image](figures/case_study_Epilepsy_task120.svg) |   ![image](figures/case_study_CharacterTrajectories_task65.svg)|
|:--:|:--:
|*Epilepsy, Task 120.*|*CharacterTrajectories, Task65.* |   


## Experimental Highlights (UCR Dataset)
<img src="figures/UEA_comparison_GRSA_vs_others_tauL0.1_all.svg" width="800">


# Getting Started

## Prerequisites

- Install Python 3.8 or higher.

- Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Code

Run the main script:

```bash
python main.py
```


## Results

- Performance analysis is stored in `analysis` directory within each benchmark folder.
- Model outputs for Regression or GRSA are saved in `results` directory within each benchmark folder.


# Main Results

## Overall Results

<img src="figures/accuracy_bar_UEA_tauL0.1_all.svg" width="800">

<img src="figures/critical_difference_diagram_tauL0.1_all.svg" width="800">


## Potential for Early Classification

<img src="figures/UEA_different-test-length.svg" width="800">


