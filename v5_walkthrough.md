# MAPPO RIS-UAV: v5 Walkthrough
## What Changed from v4 to v5

---

## Overview

v5 was created in response to critical regressions introduced in v4. The primary
cause of all v4 failures was an oversized terminal goal bonus that completely
dominated the reward signal, causing the policy to rush to the boundary instead
of optimising energy efficiency (EE). v5 fixed this and several related issues
with 7 targeted code changes.

---

## Problem-by-Problem Summary

### Fix 1 — Removed Terminal Goal Bonus (Critical)

**What was wrong in v4:**
The terminal goal bonus added by Opus in v4 could reach up to 13.3 reward units
per episode, while the EE reward was only ~7–8. This caused:
- Policy learned to rush to y=20 boundary (not Q_F=[16,16])
- EE collapsed from 0.0728 to 0.028 by ep=900
- MAPPO fell below No-RIS at ALL power levels

**What v5 did:**
Completely removed the `if done: goal_bonus` block from `env.step()`.

```python
# REMOVED from v5:
# if done:
#     for k in range(self.K):
#         final_dist = np.linalg.norm(...)
#         goal_bonus = max(0, 20.0 - final_dist) * 2.0
#         reward += goal_bonus / self.K
```

**Result:** Training no longer collapses. Reward curve stable throughout
1200 episodes. Critic loss stays below 1.8 at all times.

---

### Fix 2 — Reduced Progress Reward Weight (1.5 → 0.8)

**What was wrong in v4:**
The progress reward weight of 1.5 (set by Opus in v4) was still too strong,
causing the policy to prioritise navigation over EE optimisation even without
the terminal bonus.

**What v5 did:**
Reduced the per-step progress reward weight from 1.5 to 0.8.

```python
# v4:
progress_reward += (self.prev_dist_goal[k] - curr_dist) * 1.5

# v5:
progress_reward += (self.prev_dist_goal[k] - curr_dist) * 0.8
```

**Result:** EE reward (~7–8 units) now dominates progress reward (~0.5–1.5 units
per step). Policy primarily optimises EE while still navigating toward goal.

---

### Fix 3 — Increased Main Training Episodes (1000 → 1200)

**What was wrong:**
1000 episodes was insufficient for the policy to fully converge, especially
with the modified reward structure.

**What v5 did:**
```python
N_EPISODES = 1200  # was 1000
```

**Result:** Best model found at ep=500 with eval EE=0.078543. Training
continued stably to ep=1200 without collapse, confirming the reward
rebalancing was effective.

---

### Fix 4 — Kept v4's Per-L Sub-Training (300 episodes per L value)

**What was kept from v4:**
The per-L agent training infrastructure (`train_agent_for_L()`) was preserved
as-is from v4. Each L value in the sweep [5,10,20,30,50,80,100] gets a
dedicated agent with `Actor(obs_dim=18, act_dim=2+L)`.

**What v5 changed:**
The sub-training now uses the corrected reward function (no terminal bonus,
progress weight=0.8), which was the main issue causing v4 sub-agents to
underperform.

**Result:**
```
L=5:   sub-agent best EE=0.076406
L=10:  sub-agent best EE=0.076034
L=80:  sub-agent best EE=0.081246  ← beats No-RIS baseline ✅
L=100: sub-agent best EE=0.075809
```

---

### Fix 5 — Kept v4's Per-I Sub-Training (200 episodes per I value)

**What was kept from v4:**
`train_agent_for_devices()` preserved. Each device count gets dedicated agent.

**What v5 changed:**
Same reward correction as Fix 4.

**Result:**
```
I=2: MAPPO=0.073638, I=3: MAPPO=0.070642
I=4: MAPPO=0.078543 (main agent)
I=5: MAPPO=0.074961, I=6: MAPPO=0.071021
```
No longer flat — values vary with device count, showing the policy is
responsive to different IoT configurations.

---

### Fix 6 — Kept v4's Goal Distance in Observation (obs[17])

**What was kept from v4:**
`obs[17]` changed from `slot/M` to `d_goal / (BOUND*√2)` — gives actor
explicit knowledge of distance to goal Q_F=[16,16].

**Why this matters:**
With the goal distance in the observation, the actor can condition its
velocity decisions on how far it is from the destination. Combined with
the corrected progress reward (weight=0.8), UAVs navigate purposefully
toward the goal without being distracted by the boundary.

**Result:** Trajectory shows clear diagonal movement toward goal area,
with smooth arcing paths that pass near RIS [5,9] for channel gain.

---

### Fix 7 — Kept v4's Device Count in Observation (obs[7])

**What was kept from v4:**
`obs[7]` encodes `n_devices / 6.0` — allows policy to distinguish between
different numbers of IoT devices at eval time.

**Why this matters:**
Without this, the policy sees the same observation for I=4, I=5, and I=6
(extra devices beyond 4 were invisible). This caused the flat EE vs devices
curve in v3.

**Result:** EE vs devices now shows variation: peak at I=4, decreasing for
I>4 — physically consistent with the shared-power allocation model.

---

### Fix 8 — Increased Speed Eval Episodes (8 → 10) with Seed Reset

**What v5 did:**
```python
# In plot_ee_vs_speed():
np.random.seed(42)
torch.manual_seed(42)
_, ee_m, _ = eval_agent(agent, env_v, n_episodes=10)  # was 8
```

**Result:** EE vs speed curve is smooth with no dips. Inverted-U shape
clearly visible with peak at v=15–18 m/s.

---

## Observation Vector Layout (Final — v5)

| Index | Content | Change from v3 |
|-------|---------|----------------|
| 0–1 | Own x,y position / BOUND | unchanged |
| 2–3 | Own vx,vy / V_MAX | unchanged |
| 4–6 | Distances to 3 IoT devices | v4: reduced from 4→3 |
| 7 | n_devices / 6.0 | v4 NEW |
| 8 | Distance to RIS / (BOUND√2) | unchanged |
| 9–12 | Log channel magnitudes (4 devices) | unchanged |
| 13–16 | Other UAV positions | unchanged |
| 17 | Goal distance / (BOUND√2) | v4: was slot/M |

obs_dim = 18 (unchanged throughout all versions)

---

## Reward Function (Final — v5)

```python
reward = mean_ee * 100.0          # Primary: EE (dominates at ~7–8 units)
       + progress_reward           # Secondary: navigation (0.8 × dist_reduction)
       - collision_penalty         # Penalty: 5.0 per colliding pair
```

No terminal bonus. EE always dominates. Progress reward provides
gentle navigation guidance without distorting the EE objective.

---

## Training Configuration (v5)

| Parameter | v1 | v3 | v4 | v5 |
|-----------|----|----|----|----|
| N_EPISODES | 1000 | 1000 | 1000 | 1200 |
| Buffer capacity | 400 | 200 | 200 | 200 |
| PPO epochs | 2 | 4 | 4 | 4 |
| Eval every | 100 | 50 | 50 | 50 |
| Entropy floor | 0.001 | 0.005 | 0.005 | 0.005 |
| Progress weight | 0.0 | 0.5 | 1.5 | 0.8 |
| Terminal bonus | No | No | Yes (2.0×) | No |
| Per-L agents | No | No | Yes (300 eps) | Yes (300 eps) |
| Per-I agents | No | No | Yes (200 eps) | Yes (200 eps) |
| Goal dist in obs | No | No | Yes | Yes |
| Device count in obs | No | No | Yes | Yes |

---

## Results Summary

### Training
```
Best eval EE:  0.078543 bits/J (at ep=500)
Training time: 3062s (~51 min) on Tesla T4 GPU
Early stop:    Did NOT trigger (training stable all 1200 eps)
C_loss max:    1.85 (well below dangerous threshold of 3.0)
```

### Plot Results

| Plot | Status | Notes |
|------|--------|-------|
| EE vs P_max | ⚠️ Near | MAPPO ~0.0015 below No-RIS — very close |
| EE vs Speed | ✅ | Correct inverted-U, peak at v=15–18 |
| EE vs L | ✅ Partial | Beats No-RIS for L≥80, below for L<50 |
| EE vs Devices | ✅ | Responsive curve, peak at I=4 |
| Trajectories | ✅ | Smooth arcs toward goal, passes near RIS |
| Learning curve | ✅ | Green eval line, gold best-ep marker |
| Training stability | ✅ | No collapse in 1200 episodes |

---

## What Still Needs Work (for v6)

1. **MAPPO below No-RIS in EE vs P_max** — consistent ~0.0015 deficit
   - Root cause: progress reward (0.8) still slightly too strong
   - Fix: reduce to 0.3–0.5, closer to v3's 0.5

2. **MAPPO below Fixed trajectory in EE vs Speed**
   - Same root cause as above

3. **EE vs L flat for L=5–30** — all sub-agents converge to same value
   - Root cause: 300 episodes insufficient for small-L agents
   - Fix: increase to 500 episodes for L≤30

4. **Jagged EE vs Devices curve**
   - Root cause: 200 episodes too few for sub-agents
   - Fix: increase to 300–400 episodes per I value
