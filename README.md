Frankenstein: Evolving Agents World

Summary
- A NumPy-only agent‑based simulation that explores emergent coordination under resource constraints. Agents live on a toroidal grid with renewable resources and hazards, control themselves with tiny Elman RNNs, communicate via spatial vector fields, and reproduce sexually. The system includes hierarchical “macro” options that mutate structurally, light logging, and clean configurability.

Highlights
- First‑principles design: no external RL frameworks. Everything from sensing, action heads, energy accounting, reproduction, spatial comms, to logging is implemented directly in Python/NumPy.
- Tiny recurrent controllers: Elman RNN with separate heads for action logits and continuous parameters (velocity, transfer delta, emission radius/vector, channel selection) — compact, inspectable, and fast.
- Spatial communication fields: continuous vector fields per channel with decay and local broadcast disks, enabling cheap, local coordination without global message passing.
- Hierarchical options: short, mutable action “macros” with guards; the controller chooses between primitives and macros, akin to options in hierarchical RL.
- Evolutionary operators: sexual recombination of controllers via weight averaging + noise, genome‑level mutation of capacities and macros, and asexual cloning when configured.
- Efficient neighborhood access: per‑cell spatial buckets to scan local neighborhoods in O(k) around each agent, rather than naive all‑pairs.
- Reproducibility and cleanliness: dataclass configs, typed code, fixed RNG seed, and JSON logs for analysis.

Tech Stack
- Language: Python 3.9–3.12
- Libraries: NumPy, dataclasses, typing (no PyTorch/TF)
- Algorithms/Models: Elman RNN controller; Holling Type‑II harvest dynamics; softmax policy with continuous parameter heads; spatial hashing via cell buckets; message‑field decay dynamics; neuroevolution with structural macro mutations and sexual recombination.

Run It
- Install: pip install numpy
- Execute: python project_frankie.py
- Output: writes run_summary.json with periodic snapshots and prints concise tick‑level summaries.

Configuration
- Tweak dataclass defaults near the top of the file:
  - World/grid: world size and torus wrapping
  - Resources: carrying capacity K, regrowth r, patchiness, regrowth_mode ('constant'|'logistic')
  - Physics: movement/harvest costs, basal metabolism, comm costs, injury risk
  - Communication: channels, bandwidth (vector dims), decay, r_max
  - Controller: hidden size, init scale
  - Macros: count, length bounds, mutation rates
  - Reproduction: mode ('sexual'|'asexual'), thresholds, mate radius/window
  - Evolution: population size, mutation sigma
  - Run: ticks, log interval, rng seed

Code Tour
- World: project_frankie.py:164 — toroidal grid, patchy resource init via iterative blur, logistic or constant regrowth, Bernoulli hazards, and message‑field decay.
- Macros: project_frankie.py:257 — compact sequence options with per‑step params and energy guards.
- Controller: project_frankie.py:323 — TinyRNNController with heads for action logits and parameters; softmax policy over primitives + macro slots.
- Agent: project_frankie.py:464 — energy homeostasis, sensing (local means/gradients and message vectors), primitives (move/harvest/transfer/emit/mate), and basal metabolism.
- Population/Scheduler: project_frankie.py:647 — bucketed neighbor scans, action execution, and sexual/asexual reproduction resolution.
- Logging: project_frankie.py:1059 — lightweight JSON snapshots; entrypoint below.
- Entrypoint: project_frankie.py:1098 run_demo(); project_frankie.py:1126 if __name__ == "__main__":

Of Interest:
- Energy‑centric physics: all actions pay explicit costs; communication has bandwidth, radius, and signal‑power costs, enforcing trade‑offs between movement, harvesting, messaging, and reproduction.
- Local comm substrate: agents write/read continuous vectors into spatial fields, which decay over time — a cheap medium that still allows coordination signals, gradients, and stigmergy‑like behavior.
- Hierarchical behavior: macros let evolution discover reusable multi‑step patterns; the controller arbitrates between primitives and these higher‑level options.
- Neuroevolution basics, cleanly done: controller weights mutate with Gaussian noise; sexual recombination averages parent weights + noise; macros undergo birth/death and micro‑perturbations.

Outputs
- run_summary.json contains a history list with:
  - t: tick number
  - pop_size: alive agents
  - mean_energy: average energy
  - births/deaths: deltas since last snapshot
  - messages: change in total message‑field magnitude
  - harvest: resource consumed during the interval

Quick Tweaks
- Faster demo: lower RunConfig.ticks
- More agents: raise EvolutionConfig.pop_size
- Easier world: increase ResourceConfig.K or reduce hazards
- Asexual only: set ReproConfig.mode='asexual'

Potential Extensions
- Replace Elman with GRU/LSTM while keeping heads and APIs
- Richer sensors (e.g., multi‑cell patches, obstacle maps)
- Visualization of trajectories and message fields
- Fitness shaping or tasks layered over the world (e.g., target seeking, rendezvous)

Notes
- This code is intentionally compact and hackable. It favors clarity of mechanisms (energy budgets, local comms, minimal controllers) over heavy frameworks, making it easy to modify during research sprints or interviews.
