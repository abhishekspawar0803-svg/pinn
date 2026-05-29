# Physics-Informed Neural Networks for Circuit Dynamics: A Comparative Analysis

A PyTorch project comparing a standard multilayer perceptron (MLP) and a physics-informed neural network (PINN) for learning the dynamics of a parallel RLC circuit from synthetic data.

This repository is built around a simple research question: does adding circuit physics into the training objective improve prediction quality compared with a purely data-driven model?

## Project Overview

The project presents a controlled comparison between two models trained on the same synthetic parallel RLC dataset:

- **Baseline MLP** trained using supervised mean squared error
- **PINN** trained using supervised loss plus physics-based residual penalties

Both models use the same feedforward architecture so that the comparison focuses on the impact of the loss formulation rather than model size.

The final experiment uses:

- **Inputs:** `time, R, L, C, vs`
- **Outputs:** `IC, IL`
- **Framework:** PyTorch
- **Derivative computation:** `torch.autograd.grad`

## Why This Project Matters

Physics-Informed Neural Networks are often presented as a more principled alternative to standard neural networks. But in practice, their usefulness depends on the data regime and the physical constraints being enforced.

This project is valuable because it does not assume that the PINN must perform better. Instead, it evaluates both models fairly on:

- predictive accuracy
- physics consistency
- training behavior

The final result is a meaningful negative result: in this setup, the standard baseline outperformed the PINN.

## Circuit Physics Used

The PINN incorporates circuit equations directly into the loss function.

The physics residuals are based on:

- **Kirchhoff’s Current Law (KCL)**  
  \( i_C + i_L + i_R = 0 \)

- **Capacitor relation**  
  \( i_C = C \frac{dv}{dt} \)

Using autograd, the notebook differentiates network-derived quantities with respect to time to compute residual terms. This turns the model into more than a standard regressor; it becomes a constrained learning system informed by circuit behavior.

## Methodology

### 1. Data generation

Synthetic trajectories were generated for a parallel RLC circuit over multiple parameter settings and source values. The dataset is included in the repository to keep the experiment reproducible.

### 2. Baseline model

A standard MLP was trained using only supervised mean squared error between predicted and true currents.

### 3. PINN model

A second model with the same architecture was trained using:

- data loss
- KCL residual loss
- capacitor residual loss

The physics term was weighted progressively during training instead of applying full physics regularization from the beginning.

### 4. Residual analysis

The project also evaluates whether the PINN actually improves physical consistency by examining residual behavior on trajectories, not just prediction error.

## Model Architecture

Both the baseline and PINN use the same network:

- Linear(5 → 64)
- Tanh
- Linear(64 → 32)
- Tanh
- Linear(32 → 15)
- Tanh
- Linear(15 → 2)

This keeps the comparison fair and isolates the effect of physics-informed training.

## Final Result

In the final notebook run:

- **Baseline test MSE:** `0.00015775408489086354`
- **PINN test MSE:** `0.22964280067632595`

### Key takeaway

In this dense, noise-free synthetic setting, the baseline MLP achieved substantially better predictive accuracy than the PINN.

This suggests that explicit physics regularization is **not automatically beneficial** when the dataset is already rich and clean. In such settings, the additional residual constraints can make optimization harder and reduce predictive performance.

This does not mean PINNs are ineffective in general. Their strengths are more likely to appear when:

- data is sparse
- measurements are noisy
- some states are unobserved
- extrapolation beyond the training regime is important

## Engineering Takeaways

This project demonstrates:

- custom PyTorch training loops
- autograd-based derivative computation inside the loss
- translation of electrical engineering equations into machine learning objectives
- fair baseline comparison under a shared architecture
- residual-based evaluation beyond plain test loss
- honest reporting of a negative experimental result

That makes the repository a useful engineering project, not just a notebook with plots.

## Repository Structure

```text
.
├── dataset/            # Synthetic training, validation, and test data
├── images/             # Training curves, residual plots, and circuit visuals
├── .gitattributes
├── .gitignore
├── PINN.ipynb          # Final comparative PINN vs baseline notebook
├── README.md
├── circuit_pinn.png    # Circuit diagram asset
├── data_builder.ipynb  # Synthetic dataset generation notebook
└── requirements.txt
```

## Visuals

### Circuit setup and example data

**Circuit PINN setup**

![Circuit PINN setup](images/circuit_pinn.png)

**Example trajectory from the dataset**

![Example trajectory from the dataset](images/example_data_plot.png)

### Training loss comparison

**Baseline training loss**

![Baseline training loss](images/baseline_train_loss.png)

**Physics-informed training loss**

![Physics-informed training loss](images/physics_train_loss.png)

### Residual comparison

**Baseline residual behavior**

![Baseline residual behavior](images/baseline_residual.png)

**Physics-informed residual behavior**

![Physics-informed residual behavior](images/physics_residual.png)

## How to Run

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open `PINN.ipynb` and run the cells in order.

## Possible Extensions

A few natural next steps for this project are:

- test the same setup with noisy observations
- reduce training-data density and study low-data behavior
- evaluate extrapolation to unseen parameter ranges
- tune the physics-loss weighting schedule
- predict additional circuit variables such as voltage or resistor current
- compare with alternative PINN formulations or adaptive weighting methods

## Summary

This repository presents a comparative study of a standard MLP and a physics-informed neural network for parallel RLC circuit dynamics.

The main result is intentionally honest: in the final experiment, the simpler baseline performed better. That negative result is part of the value of the project because it shows engineering judgment, rigorous comparison, and careful evaluation rather than hype-driven conclusions.