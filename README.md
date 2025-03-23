# Gene Regulatory Network Simulation-Based Inference (grn_sbi)

This repository contains code for inferring parameters of a Gene Regulatory Network (GRN) model using Simulation-Based Inference (SBI) techniques.

## Overview

Gene Regulatory Networks are complex systems that control gene expression in biological organisms. This project implements a simplified repressilator model based on the classic paper by Elowitz and Leibler, which consists of a 3-gene network that produces oscillatory behavior.

The main goal of this project is to demonstrate how simulation-based inference can be used to infer the parameters of a GRN model from observed data.

## Components

- `grn.py`: Implements the Gene Regulatory Network model with a repressilator architecture
- `main.py`: Performs parameter inference using simulation-based inference techniques

## The GRN Model

The model implements a 3-gene repressilator with the following parameters:

- `alpha`: Maximum transcription rate in the absence of repressor
- `alpha0`: Leakiness of promoter (basal transcription rate)
- `beta`: Ratio of protein decay rate to mRNA decay rate
- `n`: Hill coefficient (cooperativity)
- `K`: Repression coefficient

The dynamics of the system are described by a set of ordinary differential equations (ODEs) that model the interactions between genes and proteins.

## Simulation-Based Inference

The parameter inference process follows these steps:

1. Define true parameters and prior distributions
2. Generate synthetic observed data using the true parameters
3. Run simulations with parameters sampled from the prior
4. Train a neural density estimator to approximate the posterior distribution
5. Sample from the posterior distribution
6. Analyze and visualize the results

## Usage

To run the parameter inference:

```
bash
python main.py
```

This will:
1. Generate synthetic data
2. Run simulations for training
3. Perform inference
4. Generate plots of the posterior distributions
5. Validate the inference by comparing simulations with true and inferred parameters