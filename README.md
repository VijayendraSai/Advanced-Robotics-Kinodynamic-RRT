# Kinodynamic RRT Motion Planning in Obstacle-Rich Environments

## 📌 Introduction

This project focuses on developing an effective motion planning strategy using **Kinodynamic Rapidly-exploring Random Trees (RRT)** in environments cluttered with obstacles. The goal is to generate dynamically feasible, collision-free trajectories for a point robot navigating from a start point (A) to a goal point (B). Unlike traditional geometric planning, kinodynamic RRT also respects the robot's dynamic constraints, ensuring smooth and safe navigation.

---

## 🚧 Problem Description

- A **point robot** located at origin A must reach a goal B through a corridor-like environment.
- The robot’s **state** includes its position `q ∈ ℝ²` and velocity `q̇ ∈ ℝ²`, so state `x = {q, q̇}`.
- The environment includes **n obstacles** (starting with `n = 1`), which must be avoided.
- **Controls** `u = (ux, uy) ∈ U` are applied to move the robot.
- The state space `X` is divided into:
  - Collision-free states: `Xf`
  - Obstacle states: `Xo`

The planning algorithm must build a **feasible plan** `p(T)` that, when executed, produces a valid trajectory `τ(x0, p(T)) = {x0, ..., xT}` avoiding all obstacles.

---

## 🌲 Kinodynamic RRT Algorithm

The motion planner constructs a tree by:
1. Sampling random configurations.
2. Finding the nearest neighbor in the tree.
3. Sampling a control.
4. Simulating the motion under the control.
5. Adding the node to the tree if the path is valid.
6. Checking if the goal has been reached.

### Algorithm Overview

```text
Kinodynamic-RRT(X, U, x0, XG):
1. T ← {x0}
2. while termination condition not met:
3.     x_rand ← SAMPLE-CONFIGURATION()
4.     x_near ← NEAREST-NEIGHBOR(T, x_rand)
5.     u ← SAMPLE-CONTROL()
6.     x_new ← SIMULATE(x_near, u)
7.     if (x_near → x_new) ∈ Xf:
8.         EXTEND-TREE(T, x_near → x_new)
9.         if GOAL-CHECK(x_new): return plan
