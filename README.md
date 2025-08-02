# Particle_phy_simulation
A deep PINN framework for 3D particle collision simulation with momentum conservation and interactive visualization

## Overview

This project implements a **Complex Physics-Informed Neural Network (PINN)** with deep residual connections for simulating particle-particle interactions and collisions in three dimensions. It is designed with research-grade flexibility and interactivity, especially for scientific computing and machine learning in physics.

## Features

- **Deep Residual Architecture:**
Uses an 8-layer residual network (512 units per layer, GELU activation, LayerNorm), enabling the model to learn highly complex relationships in particle collision outcomes.
- **Physics-Informed Loss:**
Incorporates explicit *momentum conservation* in the training loss, ensuring the model's predictions respect fundamental physical laws — critical for simulating realistic collisions.
- **Flexible Data Generation:**
Synthetic data generator supports both "2-way" and "3-way" particle collision scenarios.
- **Interactive 3D Visualization:**
Predicts outgoing particle momenta and visualizes them using Plotly, offering interactive rotation, zoom, and hover features to deeply analyze each collision event.
- **Modular Training Pipeline:**
Training loop supports configurable batch size, epochs, and physics loss scaling.


### 1. Requirements

```bash
pip install torch plotly numpy
```


### 2. Model Definition

```python
class ComplexPINN(nn.Module):
    def __init__(self, input_dim=6, output_dim=9):
        # ... as in provided code ...
```


### 3. Data Generation

```python
X, Y = generate_data(num_samples=1000, mode="2-way")  # or "3-way"
```


### 4. Training

```python
model = ComplexPINN()
train_model(model, X, Y, epochs=1000, batch_size=64, lambda_phys=0.2)
```


### 5. Visualization

```python
visualize_collision(model, X, sample_index=5, title="Sample Particle Collision")
```

- Change `sample_index` to see outputs for different events.


## Example: Typical Workflow

```python
X, Y = generate_data(num_samples=2000, mode="3-way")
model = ComplexPINN()
train_model(model, X, Y, epochs=500)
visualize_collision(model, X, sample_index=10, title="3-way Collision Example")
```
