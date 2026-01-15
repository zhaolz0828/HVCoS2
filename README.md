# 🤖 GNN–PPO Multi-Route Optimizer

This module implements the **GNN–PPO based multi-route optimizer** in **HVCoS**.  
It solves a **min–max mTSP** (multi-route) problem and outputs balanced USV routes 🌆🚚.

---

## ✨ Key Idea

We optimize **multiple USV routes jointly** by reinforcement learning:

- build K-NN graph ➜ GNN encoding  
- learn operator policy by PPO  
- minimize **max route length**  
- also reduce **route length imbalance**

Final output = balanced routes for USVs 👍  

---

## 🔧 What It Does

- Read map + tasks
- Build hybrid distance (road + Euclid)
- Initial clustering (angle sweep)
- Search operators:
  - intra-2opt
  - intra-Oropt
  - inter-relocate
  - inter-swap
  - ruin-recreate
- PPO decides which operator to apply
- Best result saved to json

---

## 📁 Input Files

| File | Meaning |
|---|---|
| `task_coordinate.csv` | task ID + coordinates |
| `distance_matrix.csv` | real road distance |
| `road_length.csv` | service time per node |

📌 depot = index 0 (fixed at (0,0))

---

## 🚀 Run

```bash
python Param-GPPO-TS-xxx.py
