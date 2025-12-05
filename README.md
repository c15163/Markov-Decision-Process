# Markov Decision Processes: Value Iteration, Policy Iteration, and Q-Learning

This repository provides a complete implementation and analysis of three fundamental Markov Decision Process (MDP) algorithms:

- Value Iteration (VI)
- Policy Iteration (PI)
- Q-Learning (QL)

The project evaluates these algorithms on both grid-based and non-grid MDP environments, with multiple state sizes, discount factors, learning rates, and convergence thresholds.  
All results and figures discussed in the project report are reproduced by the included Python script.

---

## Repository Structure

```
project_root/
│
├── Markov_Decision_Process.py        # Full implementation and all experiments
├── Markov_Decision_Process.pdf       # Final report containing all results and figures
└── README.md                         # Documentation (this file)
```

---

## Environments

This project evaluates MDP algorithms on two types of environments.

### 1. Frozen Lake (Grid World)

Two grid sizes are tested:

- Small grid: 5×5 (25 states)
- Large grid: 20×20 (400 states)

Transition dynamics:

- Frozen states: small negative reward (−0.01)
- Hole states: negative reward (−1)
- Goal state: positive reward (+1)

These grids are generated using `hiive.mdptoolbox.example.frozen_lake.random_map`.

### 2. Forest Management (Non-Grid MDP)

Two configurations:

- Small: 25 states  
- Large: 400 states  

Model parameters follow:

- Burn probability: p = 0.1  
- Rewards: r1 = 4 (harvest), r2 = 2 (wait)

Environment generated using `hiive.mdptoolbox.example.forest`.

---

## Algorithms Implemented

### Value Iteration (VI)
- Sweeps over all states updating value function
- Convergence analysis using epsilon thresholds
- Runtime comparison across grid sizes and discount factors

### Policy Iteration (PI)
- Alternates between policy evaluation and policy improvement
- Fewer iterations but potentially longer per-iteration compute time

### Q-Learning (QL)
- Model-free learning based on TD updates
- Learning rate (α) and exploration schedule (ε-greedy) are varied
- Comparison between convergence speed and cumulative reward

---

## Experiments

The following experimental analyses are included:

### 1. Convergence Analysis
- Value function changes over iterations
- Policy stability for PI
- Q-value convergence patterns

### 2. Reward Comparison
- Total episodic rewards across algorithms
- Sensitivity to discount factor (γ)
- Grid vs. non-grid reward performance differences

### 3. Execution Time
- VI / PI runtime scaling with number of states
- Q-Learning runtime as episodes increase

### 4. Hyperparameter Studies
- VI: epsilon threshold, discount factor
- PI: discount factor
- QL: learning rate α, exploration ε, episodes

### 5. Scalability Analysis
- Comparison between small (25 states) and large (400 states) environments
- Grid vs non-grid MDP performance

All figures referenced in the PDF report are generated automatically by the script.

---

## How to Run

Install requirements:

```bash
pip install numpy matplotlib pandas hiive-mdptoolbox
```

Run all experiments:

```bash
python Markov_Decision_Process.py
```

All plots and metrics will be saved automatically in the working directory.

---


## Summary of Findings

- Value Iteration converges reliably but may require many sweeps on large state spaces.
- Policy Iteration converges in fewer iterations but with higher per-iteration cost.
- Q-Learning performs competitively but requires sufficient exploration and careful α/ε tuning.
- Larger environments significantly amplify differences in convergence and runtime.
- Grid and non-grid MDPs display different sensitivities to γ, particularly under Q-learning.

Full interpretation, tables, and plots are available in the PDF report.

---

## Reference

See `Markov_Decision_Process.pdf` for complete analysis and visualization results.
