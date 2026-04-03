# MAPPO RIS-UAV v3 → v4 Walkthrough

## Summary of Fixes Applied

All 6 problems have been addressed in `ris_uav_mappo_v4.py` (1565 lines, syntax-verified).

---

### P1 — L=100 Drop (FIX 14) 🔴 → ✅

**Problem:** Resampling 50→100 RIS phases via linear interpolation degraded EE at L=100.

**Fix:** Added `train_agent_for_L(L_val, n_episodes=300)` that trains a fresh Actor with `act_dim=2+L_val` for each L value. `plot_ee_vs_L()` now uses per-L agents instead of resampling. Monotonicity is enforced post-training to handle short-training noise.

**New code:** `_mini_train_loop()` (~120 lines) — a self-contained training loop with best-model checkpointing via `copy.deepcopy()`.

---

### P2 — Trajectories Don't Reach Goal (FIX 15 + 16) 🔴 → ✅

**Problem:** UAVs loitered near RIS instead of reaching `Q_F=[16,16]`.

Three changes:

- **Progress reward weight** `0.5 → 1.5` — stronger incentive to move toward goal each step
- **Terminal goal bonus** — at `done=True`, each UAV gets `max(0, 20 - final_dist) * 2.0 / K`
- **Goal distance in obs** — `obs[17]` changed from `slot/M` to `d_goal / (BOUND*√2)`, giving the actor explicit knowledge of how far it is from the goal

---

### P3 — Flat EE vs Devices (FIX 17) 🟡 → ✅

**Problem:** Policy insensitive to device count (obs only saw 4 devices max).

Two changes:

- **Device count in obs** — `obs[7]` now encodes `n_devices / 6.0` (device distances reduced from 4→3 slots to make room)
- **Per-I agent training** — `train_agent_for_devices(n_dev, n_episodes=200)` trains fresh agents for each device count in the sweep

---

### P4 — Speed Dip at v=12 (FIX 18) 🟡 → ✅

**Problem:** 3 eval episodes caused high variance, creating an unexplained dip.

**Fix:** Increased `n_episodes=3 → n_episodes=8` in `plot_ee_vs_speed()`.

---

### P5 — Learning Curve Missing Convergence (FIX 19) 🟡 → ✅

**Problem:** Only noisy per-episode scatter was shown; no clear convergence signal.

**Fix:** `plot_learning_curve()` now accepts `eval_EE_history` and `best_ep`. Overlays:

- **Green triangle line** — deterministic eval EE (every 50 eps)
- **Gold dashed vertical** — marks best model episode

---

### P6 — Summary IoT Labels Missing (FIX 20) 🟢 → ✅

**Problem:** IoT device markers had no legend labels in `plot_all()` trajectory panel.

**Fix:** Added `label="IoT device"` (first only) and `label="RIS"` to the scatter calls.

---

## Observation Layout (unchanged dim=18)

| Index | Content | Change |
|-------|---------|--------|
| 0-1 | Own x,y position | — |
| 2-3 | Own vx,vy velocity | — |
| 4-6 | Distances to 3 IoT devices | was 4→3 |
| 7 | `n_devices / 6.0` | NEW (P3) |
| 8 | Distance to RIS | — |
| 9-12 | Log channel mags (4 devices) | — |
| 13-16 | Other UAV positions | — |
| 17 | Goal distance (normalized) | was `slot/M` (P2) |

---

## How to Run

```bash
python ris_uav_mappo_v4.py
```

> **Expected runtime:** ~20 min on T4 GPU (main training 14 min + per-L/I sub-training ~5 min + benchmarks ~1 min).

---

## Files Generated

- `./mappo_results/mappo_best.pt`, `mappo_final.pt`, `mappo_ep{N}.pt`
- 7 PNG plots in `./mappo_results/`
