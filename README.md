# RL-GA-HYBRID-RM

This repository solves a **Traffic-Aware Traveling Salesman Problem (TSP)** over Indian cities using:

1. **Q-Learning (Reinforcement Learning)**
2. **Genetic Algorithm (GA)**
3. **Hybrid Q-Learning + GA**

The city data is read from `Indian Cities Database.csv`, and each script asks you to select a starting city.

---

## Algorithms in this Repo

## 1) Q-Learning (`QLRL.py`)

### What it does
- Builds a city-to-city distance matrix using Haversine distance.
- Trains a Q-table over many episodes to learn good next-city transitions.
- Supports:
  - **Without traffic** optimization
  - **With dynamic traffic** optimization (traffic multipliers updated over time)

### Key idea
- Reward is negative distance, so the agent learns shorter tours.
- Uses epsilon-greedy exploration with decay.

### Best use case
- When you want a **learning-based route policy** that adapts to changing traffic over training.
- Useful as a baseline RL approach and for producing an informed route prior.

### Main outputs
- Convergence plots (with/without traffic)
- Route plots (with/without traffic)
- Distance and timing comparisons

---

## 2) Genetic Algorithm (`GA.py`)

### What it does
- Evolves a population of candidate tours using:
  - Tournament selection
  - PMX crossover
  - Adaptive mutation
  - Elitism
- Includes a traffic manager to evaluate routes under dynamic traffic.
- Runs GA in two settings:
  - **With traffic**
  - **Without traffic**

### Key idea
- Search is population-based and stochastic, so it can explore many route structures efficiently.

### Best use case
- When you want a **strong global search heuristic** for large combinatorial route spaces.
- Good when you care about robust optimization under different traffic states.

### Main outputs
- Route visualizations (with/without traffic)
- Convergence curves (with/without traffic)
- Distance/travel-time summary

---

## 3) Hybrid Q-Learning + GA (`hybrid.py`)

### What it does
- Uses Q-Learning first to learn a high-quality initial policy/route.
- Seeds GA population with Q-Learning-informed solutions.
- Applies traffic-aware GA evolution with periodic traffic updates.
- Adds Q-guided improvement during GA evolution.

### Key idea
- Combines:
  - **RL exploitation of learned transitions**
  - **GA exploration/refinement of route permutations**

### Best use case
- When you want the **best overall performance** in dynamic traffic:
  - Faster convergence than plain GA from random initialization
  - Better refinement than pure Q-Learning
- Recommended for realistic, non-stationary routing scenarios.

### Main outputs
- Q-Learning convergence plot
- Hybrid GA convergence (traffic-aware) and comparison plot
- Route visualization with traffic and without traffic
- Improvement metrics over Q-Learning baseline

---

## Quick Comparison

- **Q-Learning only**: good adaptive RL baseline, learns transition values over episodes.
- **GA only**: strong evolutionary search over full route permutations.
- **Hybrid**: best of both; typically preferred for dynamic traffic-aware TSP optimization.

---

## How to Run

Run any script directly:

- `python QLRL.py`
- `python GA.py`
- `python hybrid.py`

Each script will prompt for the start city index and then generate plots/results.

---

## Notes

- Some paths in scripts are currently hardcoded to local absolute paths (dataset/output directories).  
  Update them to paths valid in your environment before running.
- Traffic effects are simulated and time-varying to model realistic road conditions.
