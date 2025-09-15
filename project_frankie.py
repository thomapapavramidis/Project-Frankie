#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evolving Agents World (first-principles, no social priors)

- Minimal physics: energy, space, resources, hazards, communication fields
- Agents with tiny recurrent controllers (Elman RNN) mutated by neuroevolution
- Primitive actions only: move, harvest, transfer (share/steal), emit, rest, mate
- Sequence macros (options) that can mutate (add/del/perturb steps)
- Asexual or sexual reproduction with energy thresholds (sexual enabled by default)
- Metrics logging (lightweight) + a toy "main" to run a single simulation

This is a compact baseline intended for extension and experimentation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import math
import json
import random

# ---------------------------
# Utilities
# ---------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-8)

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def clipped_gaussian(shape, sigma):
    return np.random.normal(0.0, sigma, size=shape)

def rand_unit_vector2():
    theta = np.random.uniform(0, 2*np.pi)
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

# ---------------------------
# Config
# ---------------------------

@dataclass
class PhysicsConfig:
    # Movement
    c_m: float = 0.4
    alpha: float = 2.0
    v_max: float = 1.0
    # Harvest
    a: float = 0.15
    h: float = 2.0
    eta: float = 6.0
    c_h: float = 0.1
    # Communication
    k0: float = 0.02
    kM: float = 0.002
    kr: float = 0.0005
    kS: float = 0.0005
    # Basal metabolism
    b0: float = 0.2
    b_brain: float = 5e-4
    b_sense: float = 0.002
    b_chan: float = 0.002
    b_macro: float = 0.02
    E_max: float = 100.0
    # Transfer/contest
    c_T: float = 0.15
    rho: float = 0.9
    p_inj: float = 0.2
    L_injury: float = 0.8

@dataclass
class ResourceConfig:
    K: float = 40.0
    r: float = 0.10
    patchiness: float = 0.6   # 0 uniform, 1 highly patchy (used for init)
    regrowth_mode: str = "constant"   # "constant" or "logistic"
    regrowth_flow: float = 0.3        # per-cell inflow per tick (used when mode="constant")

@dataclass
class HazardConfig:
    background_p: float = 0.1  # per-tick Bernoulli chance
    background_D: float = 4.0  # damage if hit
    # Future: storms, spatial fields, etc.

@dataclass
class ReproConfig:
    mode: str = "sexual"      # "sexual" or "asexual"
    E_rep: float = 1.0       # threshold to be ready
    offspring_energy_frac: float = 0.5 # fraction of E_rep given to offspring
    gestation_leak: float = 0.0        # not used here
    mate_radius: float = 3.5
    mate_intent_window: int = 10  # ticks to remember intent

@dataclass
class CommConfig:
    channels: int = 2
    bandwidth: int = 4        # M dims per channel vector field
    decay: float = 0.9        # per tick decay multiplier in field (0..1)
    r_max: float = 3.0        # max broadcast radius

@dataclass
class ControllerConfig:
    obs_size: int = 32     # will be derived at runtime (placeholder upper bound)
    hidden_size: int = 16
    param_head_size: int = 2 + 1 + 1 + 4  # [vx,vy], transfer_delta, emit_radius, emit_vec(M=4) ; channel logits will be separate
    init_weight_scale: float = 0.1
    max_params: int = 2000

@dataclass
class MacroConfig:
    init_count: int = 1
    length_min: int = 2
    length_max: int = 5
    tau_max: int = 6
    birth_rate: float = 0.05   # probability per offspring to add a brand-new macro
    death_rate: float = 0.02   # probability per macro to be deleted in offspring
    param_sigma: float = 0.1

@dataclass
class EvolutionConfig:
    pop_size: int = 30
    mutation_sigma: float = 0.05  # weight noise
    capacity_mut_sigma: float = 0.2
    structural_macro_rate: float = 0.1 # chance to mutate macro structure
    seeds: List[int] = field(default_factory=lambda: [123])

@dataclass
class WorldConfig:
    grid_size: Tuple[int, int] = (48, 48) # H, W
    torus: bool = True

@dataclass
class RunConfig:
    ticks: int = 5000
    log_interval: int = 50
    rng_seed: int = 12345

@dataclass
class Config:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    hazards: HazardConfig = field(default_factory=HazardConfig)
    reproduction: ReproConfig = field(default_factory=ReproConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    macros: MacroConfig = field(default_factory=MacroConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    run: RunConfig = field(default_factory=RunConfig)

# ---------------------------
# World
# ---------------------------

class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        H, W = cfg.world.grid_size
        # Resources & hazards
        self.R = np.zeros((H, W), dtype=np.float32)
        self.H_t = np.zeros_like(self.R)  # last harvest deducted
        self.hazard_field = np.zeros_like(self.R)  # placeholder for spatial hazards (unused for now)
        # Communication fields: (channels, bandwidth, H, W)
        self.message_field = np.zeros((cfg.comm.channels, cfg.comm.bandwidth, H, W), dtype=np.float32)
        self.rng = np.random.default_rng(cfg.run.rng_seed)
        # Initialize resources with patchiness
        self._init_resources_patchy(cfg.resources.patchiness, cfg.resources.K)
        self.tick = 0
        

    def _init_resources_patchy(self, patchiness: float, K: float):
        H, W = self.R.shape
        # Simple patchy noise via blurred random seeds
        base = self.rng.random((H, W))
        # blur using cheap neighbor averaging repeated depending on patchiness
        repeats = int(1 + patchiness * 10)
        arr = base
        for _ in range(repeats):
            arr = 0.2*arr + 0.2*np.roll(arr,1,0) + 0.2*np.roll(arr,-1,0) + 0.2*np.roll(arr,1,1) + 0.2*np.roll(arr,-1,1)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        self.R[:, :] = (0.2 + 0.8*arr) * K  # between 0.2K and K

    def wrap_pos(self, pos: np.ndarray) -> np.ndarray:
        if self.cfg.world.torus:
            H, W = self.cfg.world.grid_size
            pos[0] %= H
            pos[1] %= W
        else:
            pos[0] = np.clip(pos[0], 0, self.cfg.world.grid_size[0]-1)
            pos[1] = np.clip(pos[1], 0, self.cfg.world.grid_size[1]-1)
        return pos

    def step_resources(self):
        mode = getattr(self.cfg.resources, "regrowth_mode", "logistic")
        K = self.cfg.resources.K

        if mode == "constant":
            g = self.cfg.resources.regrowth_flow  # constant per-cell inflow
            # add inflow, subtract harvest, clamp to [0, K]
            self.R += g
            self.R -= self.H_t
            np.clip(self.R, 0.0, K, out=self.R)
        else:
            r = self.cfg.resources.r
            self.R += r * self.R * (1.0 - self.R / K)
            self.R -= self.H_t
            np.clip(self.R, 0.0, K, out=self.R)

        self.H_t.fill(0.0)  # reset for next cycle

    def step_hazards(self, agents: List["Agent"]):
        # Background hazards as independent Bernoulli for each agent
        p = self.cfg.hazards.background_p
        D = self.cfg.hazards.background_D
        if p <= 0.0 or D <= 0.0:
            return
        for ag in agents:
            if not ag.alive: 
                continue
            if np.random.random() < p:
                ag.energy -= D
                ag.hazard_damage += D

    def decay_messages(self):
        self.message_field *= self.cfg.comm.decay

    def deposit_message(self, channel: int, vec: np.ndarray, center: np.ndarray, radius: float):
        # Uniform deposit in a disk (radius in cells); add vector to each cell.
        H, W = self.cfg.world.grid_size
        r = int(max(1, round(radius)))
        cx, cy = int(round(center[0])), int(round(center[1]))
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    x, y = cx + dx, cy + dy
                    if self.cfg.world.torus:
                        x %= H
                        y %= W
                    elif not (0 <= x < H and 0 <= y < W):
                        continue
                    self.message_field[channel, :, x, y] += vec

# ---------------------------
# Macros (sequence options)
# ---------------------------

@dataclass
class MacroStep:
    # action_type: 0=rest, 1=move, 2=harvest, 3=transfer, 4=emit, 5=mate
    action_type: int
    duration: int
    # Parameters for different actions
    move_v: Tuple[float, float] = (0.0, 0.0)   # for move
    transfer_delta: float = 0.0               # energy amount to attempt (+ take, - give)
    emit_channel: int = 0
    emit_radius: float = 1.0
    emit_vec: List[float] = field(default_factory=lambda: [0.0,0.0,0.0,0.0])
    # Guard: abort if energy below threshold
    guard_E_min: float = -1.0  # -1 => disabled

@dataclass
class Macro:
    steps: List[MacroStep] = field(default_factory=list)
    current_idx: int = 0
    remaining: int = 0
    running: bool = False

    def reset(self):
        self.current_idx = 0
        self.remaining = 0
        self.running = False

    def start(self):
        if not self.steps:
            self.running = False
            return
        self.current_idx = 0
        self.remaining = max(1, self.steps[0].duration)
        self.running = True

    def step(self, agent: "Agent", world: World) -> Tuple[int, Dict[str, Any], bool]:
        """Return action_type, params, done_flag"""
        if not self.running or not self.steps:
            return 0, {}, True
        st = self.steps[self.current_idx]
        # Guard: abort if energy too low
        if st.guard_E_min >= 0 and agent.energy < st.guard_E_min:
            self.running = False
            return 0, {}, True
        self.remaining -= 1
        params: Dict[str, Any] = {}
        if st.action_type == 1:
            params["move_v"] = np.array(st.move_v, dtype=np.float32)
        elif st.action_type == 3:
            params["transfer_delta"] = float(st.transfer_delta)
        elif st.action_type == 4:
            params["emit_channel"] = int(st.emit_channel)
            params["emit_radius"] = float(st.emit_radius)
            params["emit_vec"] = np.array(st.emit_vec, dtype=np.float32)
        # when remaining hits 0, advance or finish
        if self.remaining <= 0:
            self.current_idx += 1
            if self.current_idx >= len(self.steps):
                self.running = False
                return st.action_type, params, True
            else:
                self.remaining = max(1, self.steps[self.current_idx].duration)
        return st.action_type, params, False

# ---------------------------
# Controller (tiny Elman RNN with param heads)
# ---------------------------

class TinyRNNController:
    def __init__(self, obs_size: int, hidden_size: int, action_space: int, channels: int, bandwidth: int, init_scale: float):
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_space = action_space
        self.channels = channels
        self.bandwidth = bandwidth
        # Weights
        rs = init_scale
        self.Wxh = np.random.randn(hidden_size, obs_size) * rs
        self.Whh = np.random.randn(hidden_size, hidden_size) * rs
        self.bh  = np.zeros((hidden_size, ))
        # Heads
        self.Who = np.random.randn(action_space, hidden_size) * rs  # action logits (including macro-call slots)
        self.bo  = np.zeros((action_space, ))
        # param heads: [vx, vy, transfer_delta, emit_radius, emit_vec(M)]
        param_size = 2 + 1 + 1 + bandwidth
        self.Wparam = np.random.randn(param_size, hidden_size) * rs
        self.bparam = np.zeros((param_size, ))
        # channel selection logits head
        self.Wchan = np.random.randn(self.channels, hidden_size) * rs
        self.bchan = np.zeros((self.channels, ))
        # Hidden state (per agent)
        self.h = np.zeros((hidden_size,), dtype=np.float32)

    def clone(self) -> "TinyRNNController":
        c = TinyRNNController(self.obs_size, self.hidden_size, self.action_space, self.channels, self.bandwidth, 0.0)
        c.Wxh = self.Wxh.copy(); c.Whh = self.Whh.copy(); c.bh = self.bh.copy()
        c.Who = self.Who.copy(); c.bo = self.bo.copy()
        c.Wparam = self.Wparam.copy(); c.bparam = self.bparam.copy()
        c.Wchan = self.Wchan.copy(); c.bchan = self.bchan.copy()
        c.h = np.zeros_like(self.h)
        return c

    def mutate(self, sigma: float):
        for arr in [self.Wxh, self.Whh, self.bh, self.Who, self.bo, self.Wparam, self.bparam, self.Wchan, self.bchan]:
            arr += np.random.normal(0.0, sigma, size=arr.shape)

    def forward(self, obs: np.ndarray) -> Dict[str, Any]:
        # Recurrent update
        h = tanh(self.Wxh @ obs + self.Whh @ self.h + self.bh)
        self.h = h.astype(np.float32)
        # Heads
        action_logits = self.Who @ h + self.bo
        action_probs = softmax(action_logits)
        params_raw = self.Wparam @ h + self.bparam
        # Parse params
        vx, vy = np.tanh(params_raw[0]), np.tanh(params_raw[1])
        transfer_delta = 5.0 * np.tanh(params_raw[2])   # scale to reasonable attempt
        emit_radius = 1.0 + 2.0 * sigmoid(params_raw[3]) # [1,3]
        emit_vec = np.tanh(params_raw[4:4+self.bandwidth])
        chan_logits = self.Wchan @ h + self.bchan
        chan_probs = softmax(chan_logits)
        return {
            "action_probs": action_probs,
            "vxvy": np.array([vx, vy], dtype=np.float32),
            "transfer_delta": float(transfer_delta),
            "emit_radius": float(emit_radius),
            "emit_vec": emit_vec.astype(np.float32),
            "chan_probs": chan_probs
        }

# ---------------------------
# Genome / Agent
# ---------------------------

@dataclass
class Genome:
    # capacities
    sensor_radius: float = 2.0
    movement_eff: float = 1.0  # scales movement cost (lower => cheaper)
    comm_bandwidth: int = 4
    comm_radius_max: float = 3.0
    # controller + macros
    controller: TinyRNNController = None
    macros: List[Macro] = field(default_factory=list)
    # mutation rates
    mut_sigma: float = 0.05
    macro_param_sigma: float = 0.1

    def clone_mutated(self) -> "Genome":
        g = Genome(
            sensor_radius=max(1.0, self.sensor_radius + np.random.normal(0.0, 0.2)),
            movement_eff=max(0.6, self.movement_eff + np.random.normal(0.0, 0.05)),
            comm_bandwidth=max(1, int(round(self.comm_bandwidth + np.random.normal(0.0, 0.5)))),
            comm_radius_max=max(1.0, self.comm_radius_max + np.random.normal(0.0, 0.2)),
            controller=self.controller.clone(),
            macros=[Macro(steps=[MacroStep(**vars(s)) for s in m.steps]) for m in self.macros],
            mut_sigma=max(0.01, self.mut_sigma + np.random.normal(0.0, 0.005)),
            macro_param_sigma=max(0.02, self.macro_param_sigma + np.random.normal(0.0, 0.01))
        )
        g.controller.mutate(self.mut_sigma)
        # macro birth/death
        # delete
        kept_macros = []
        for m in g.macros:
            if np.random.random() < 0.02 and len(g.macros) > 0:
                continue
            kept_macros.append(m)
        g.macros = kept_macros
        # maybe add a fresh macro
        if np.random.random() < 0.05 and len(g.macros) < 4:
            g.macros.append(random_macro())
        # micro-perturb steps
        for m in g.macros:
            for st in m.steps:
                if np.random.random() < 0.2:
                    st.duration = max(1, int(round(st.duration + np.random.normal(0,1)))) 
                if np.random.random() < 0.2:
                    st.move_v = (float(st.move_v[0] + np.random.normal(0,0.2)), float(st.move_v[1] + np.random.normal(0,0.2)))
                if np.random.random() < 0.2:
                    st.transfer_delta += np.random.normal(0, 0.5)
                if np.random.random() < 0.2:
                    st.emit_radius = max(1.0, st.emit_radius + np.random.normal(0,0.2))
                if np.random.random() < 0.2:
                    st.emit_vec = [float(x + np.random.normal(0,0.2)) for x in st.emit_vec]
        return g

def random_macro_step(cfg_comm_bandwidth: int) -> MacroStep:
    action_type = np.random.choice([0,1,2,3,4,5], p=[0.2,0.35,0.2,0.1,0.1,0.05])
    duration = int(np.random.geometric(0.4))
    duration = int(np.clip(duration, 1, 6))
    move_v = (float(np.random.uniform(-1,1)), float(np.random.uniform(-1,1)))
    transfer_delta = float(np.random.uniform(-3,3))
    emit_channel = int(np.random.randint(0, 2))
    emit_radius = float(np.random.uniform(1,3))
    emit_vec = [float(x) for x in np.random.normal(0, 0.5, size=(cfg_comm_bandwidth,))]
    guard_E_min = float(np.random.choice([-1, np.random.uniform(5,50)], p=[0.7,0.3]))
    return MacroStep(action_type, duration, move_v, transfer_delta, emit_channel, emit_radius, emit_vec, guard_E_min)

def random_macro(cfg_comm_bandwidth: int = 4) -> Macro:
    L = int(np.clip(np.random.geometric(0.5), 2, 5))
    steps = [ random_macro_step(cfg_comm_bandwidth) for _ in range(L) ]
    return Macro(steps=steps)

# ---------------------------
# Agent
# ---------------------------

AG_ACTIONS = ["rest","move","harvest","transfer","emit","mate"]  # indices 0..5

class Agent:
    _next_id = 0
    def __init__(self, pos: np.ndarray, energy: float, genome: Genome, cfg: Config, lineage_id: Optional[int]=None, parent_id: Optional[int]=None):
        self.id = Agent._next_id; Agent._next_id += 1
        self.lineage_id = lineage_id if lineage_id is not None else self.id
        self.parent_id = parent_id
        self.cfg = cfg
        self.pos = pos.astype(np.float32)
        self.energy = float(energy)
        self.age = 0
        self.alive = True
        self.children_count = 0
        self.hazard_damage = 0.0
        self.last_energy = self.energy
        # genome/controller/macros
        self.genome = genome
        # active macro
        self.active_macro_idx: Optional[int] = None
        self.active_macro_done: bool = True

    def obs_vector(self, world: World, neighbors: List["Agent"]) -> np.ndarray:
        cfgw = world.cfg.world
        H, W = cfgw.grid_size
        torus = cfgw.torus
        x = int(self.pos[0]) % H
        y = int(self.pos[1]) % W

        # local mean over a tiny fixed window based on integer radius
        rad = int(round(self.genome.sensor_radius))
        # clamp rad to e.g. 1..3 during dev to bound cost (optional)
        rad = max(1, min(rad, 3))

        # fast wrapped index helper
        def wrap(i, n):
            return i % n if torus else min(max(i, 0), n - 1)

        # accumulate mean without building big index arrays
        s = 0.0; c = 0
        for dx in range(-rad, rad + 1):
            xi = wrap(x + dx, H)
            row = world.R[xi]
            for dy in range(-rad, rad + 1):
                yi = wrap(y + dy, W)
                s += row[yi]
                c += 1
        R_mean = s / c

        # gradient (2-point stencil)
        Rx1 = world.R[wrap(x + 1, H), y]
        Rx0 = world.R[wrap(x - 1, H), y]
        Ry1 = world.R[x, wrap(y + 1, W)]
        Ry0 = world.R[x, wrap(y - 1, W)]
        gradx = 0.5 * (Rx1 - Rx0)
        grady = 0.5 * (Ry1 - Ry0)

        # neighbors summary stays as you had it
        n_count = float(len(neighbors))
        avg_e = 0.0 if n_count == 0 else float(np.mean([nb.energy for nb in neighbors]))

        # message features (already O(1))
        msg_feats = []
        for cidx in range(world.cfg.comm.channels):
            vec = world.message_field[cidx, :, x, y]
            msg_feats.extend(vec.tolist())

        dE_dt = float(self.energy - self.last_energy)
        self.last_energy = self.energy
        obs = np.array(
            [R_mean, gradx, grady, n_count, avg_e, self.energy, dE_dt, float(self.age / (1 + 2000))],
            dtype=np.float32
        )
        obs_full = np.concatenate([obs, np.array(msg_feats, dtype=np.float32)])
        min_len = max(8 + world.cfg.comm.channels * world.cfg.comm.bandwidth, 16)
        if obs_full.shape[0] < min_len:
            obs_full = np.pad(obs_full, (0, min_len - obs_full.shape[0]))
        else:
            obs_full = obs_full[:min_len]
        return obs_full.astype(np.float32)

    def basal_metabolic_cost(self) -> float:
        p = self.cfg.physics
        # approximate params count of controller
        ctrl_params = sum(arr.size for arr in [
            self.genome.controller.Wxh, self.genome.controller.Whh, self.genome.controller.bh,
            self.genome.controller.Who, self.genome.controller.bo,
            self.genome.controller.Wparam, self.genome.controller.bparam,
            self.genome.controller.Wchan, self.genome.controller.bchan
        ])
        phi_sense = self.genome.sensor_radius
        phi_chan = self.genome.comm_bandwidth * (self.genome.comm_radius_max**2)
        return (p.b0 + p.b_brain*ctrl_params + p.b_sense*phi_sense + p.b_chan*phi_chan + p.b_macro*len(self.genome.macros))

    # --- Action primitives ---

    def act_move(self, v: np.ndarray, world: World) -> float:
        # v in [-1,1]^2 scaled by v_max
        v = np.clip(v, -1, 1)
        speed = np.linalg.norm(v) * world.cfg.physics.v_max
        cost = world.cfg.physics.c_m * (speed ** world.cfg.physics.alpha)
        # update position
        new_pos = self.pos + v * world.cfg.physics.v_max
        self.pos = world.wrap_pos(new_pos)
        return cost / max(1e-6, self.genome.movement_eff)

    def act_harvest(self, world: World) -> float:
        # Cell coordinates
        x = int(round(self.pos[0])) % world.cfg.world.grid_size[0]
        y = int(round(self.pos[1])) % world.cfg.world.grid_size[1]

        # Local stock and params
        R   = world.R[x, y]
        a   = self.cfg.physics.a
        h   = self.cfg.physics.h
        eta = self.cfg.physics.eta
        c_h = self.cfg.physics.c_h

        # Holling II: raw resource captured per tick (resource units, NOT energy)
        raw = (a * R) / (1.0 + a * h * R)

        # Can't take more than what's there
        take = min(R, raw)

        # Energy gained is converted from raw by eta
        self.energy += eta * take

        # Remove only raw resource from the cell
        world.H_t[x, y] += take

        # The handling/effort cost is paid by the scheduler once
        return c_h

    def find_nearest_neighbor(self, neighbors: List["Agent"], max_dist: float=1.5) -> Optional["Agent"]:
        best = None; best_d2 = 1e9
        for nb in neighbors:
            if not nb.alive or nb.id == self.id:
                continue
            d2 = float((nb.pos[0]-self.pos[0])**2 + (nb.pos[1]-self.pos[1])**2)
            if d2 < best_d2 and d2 <= max_dist**2:
                best = nb; best_d2 = d2
        return best

    def act_transfer(self, neighbors: List["Agent"], delta: float) -> float:
        # Positive delta => attempt to take; negative => give
        target = self.find_nearest_neighbor(neighbors, max_dist=1.5)
        if target is None:
            return self.cfg.physics.c_T  # wasted attempt
        cost_attempt = self.cfg.physics.c_T
        if delta < 0.0:
            # share (give)
            amount = min(self.energy, -delta)
            self.energy -= amount
            target.energy += self.cfg.physics.rho * amount
            return cost_attempt
        else:
            # steal (take), with injury risk
            amount = min(target.energy, delta)
            target.energy -= amount
            self.energy += self.cfg.physics.rho * amount
            if np.random.random() < self.cfg.physics.p_inj:
                self.energy -= self.cfg.physics.L_injury
            return cost_attempt

    def act_emit(self, world: World, channel: int, vec: np.ndarray, radius: float) -> float:
        # cost
        k0 = world.cfg.physics.k0; kM = world.cfg.physics.kM; kr = world.cfg.physics.kr; kS = world.cfg.physics.kS
        bandwidth = vec.shape[0]
        cost = k0 + kM*bandwidth + kr*(math.pi*radius*radius) + kS*float(np.dot(vec, vec))
        # deposit
        r = min(radius, self.genome.comm_radius_max)
        world.deposit_message(channel, vec, self.pos, r)
        return cost

    def act_mate(self, neighbors: List["Agent"], world: World) -> float:
        # Sexual reproduction: both partners must have E >= E_rep/2 and be willing this tick.
        # We implement "willing" as: both selected mate action this tick and are within range.
        # The coordination is resolved externally in the scheduler. Here we just return small intent cost.
        # We'll use a flag set by the scheduler.
        return 0.02  # tiny cost for mating attempt

# ---------------------------
# Population & Scheduler
# ---------------------------

class Population:
    def __init__(self, cfg: Config, world: World):
        self.cfg = cfg
        self.world = world
        self.agents: List[Agent] = []
        self.intent_mate: Dict[int, int] = {}  # agent.id -> bool for this tick

    def add_agent(self, ag: Agent):
        self.agents.append(ag)

    def remove_dead(self):
        alive = []
        for ag in self.agents:
            if ag.alive and ag.energy > 0.0 and ag.energy < self.cfg.physics.E_max:
                alive.append(ag)
            else:
                ag.alive = False
        self.agents = alive

    def neighbors_of(self, ag: Agent) -> List[Agent]:
        H, W = self.cfg.world.grid_size
        # search radius in cells (cap to something reasonable)
        r = int(math.ceil(max(2.0, ag.genome.sensor_radius + 1.5)))
        x0 = int(ag.pos[0]) % H
        y0 = int(ag.pos[1]) % W
        res_idx = []
        for dx in range(-r, r + 1):
            xi = (x0 + dx) % H if self.cfg.world.torus else x0 + dx
            if not (0 <= xi < H):
                continue
            for dy in range(-r, r + 1):
                yj = (y0 + dy) % W if self.cfg.world.torus else y0 + dy
                if not (0 <= yj < W):
                    continue
                res_idx.extend(self._buckets[xi][yj])
        # map back to Agent objects and do a final distance check
        out = []
        for idx in res_idx:
            other = self.agents[idx]
            if not other.alive or other.id == ag.id:
                continue
            # torus-aware distance
            dx = abs(other.pos[0] - ag.pos[0]); dy = abs(other.pos[1] - ag.pos[1])
            if self.cfg.world.torus:
                dx = min(dx, H - dx); dy = min(dy, W - dy)
            if dx*dx + dy*dy <= (r*r):
                out.append(other)
        return out

    def step(self):
        
        # prune intents older than the allowed window
        t = self.world.tick
        window = self.cfg.reproduction.mate_intent_window
        for aid, t0 in list(self.intent_mate.items()):
            if t - t0 > window:
                del self.intent_mate[aid]

        cfg = self.cfg
        world = self.world


        # 1) Hazards & Resource regrowth & message decay
        world.step_hazards(self.agents)
        world.step_resources()
        world.decay_messages()


        # ---- build spatial buckets for this tick ----
        H, W = self.cfg.world.grid_size
        buckets = [[[] for _ in range(W)] for _ in range(H)]
        for idx, ag in enumerate(self.agents):
            if not ag.alive:
                continue
            x = int(ag.pos[0]) % H
            y = int(ag.pos[1]) % W
            buckets[x][y].append(idx)
        self._buckets = buckets  # store for neighbors_of


        # 2) Agents act
        # We'll collect mating intents; after all have acted we resolve successful matings.
        for ag in self.agents:
            if not ag.alive:
                continue
            ag.age += 1

            # Basal metabolism
            bmr = ag.basal_metabolic_cost()
            ag.energy -= bmr

            # Sense
            neighbors = self.neighbors_of(ag)
            obs = ag.obs_vector(world, neighbors)

            # Decide: macro vs base controller
            action_type = 0
            params: Dict[str, Any] = {}

            # If macro running, step it
            if ag.active_macro_idx is not None and not ag.active_macro_done:
                macro = ag.genome.macros[ag.active_macro_idx]
                a_type, prms, done = macro.step(ag, world)
                action_type = a_type
                params = prms
                ag.active_macro_done = done
                if done:
                    ag.active_macro_idx = None
            else:
                # Use controller
                out = ag.genome.controller.forward(obs)
                probs = out["action_probs"]
                # total action space = 6 primitives + len(macros) macro calls
                num_macros = len(ag.genome.macros)
                # If controller only knows a fixed action space, we restrict to 6 + num_macros by slicing:
                A = min(len(probs), 6 + num_macros)
                probs = probs[:A]
                s = np.sum(probs)
                if s < 1e-8:
                    probs = np.ones(A) / A  # fallback to uniform
                else:
                    probs /= s
                a_idx = int(np.random.choice(np.arange(A), p=probs))
                if a_idx < 6:
                    action_type = a_idx
                    params = {
                        "move_v": out["vxvy"],
                        "transfer_delta": out["transfer_delta"],
                        "emit_radius": out["emit_radius"],
                        "emit_vec": out["emit_vec"],
                        "emit_channel": int(np.argmax(out["chan_probs"]))
                    }
                else:
                    # call macro (a_idx - 6)
                    m_idx = a_idx - 6
                    if 0 <= m_idx < num_macros:
                        ag.active_macro_idx = m_idx
                        ag.active_macro_done = False
                        ag.genome.macros[m_idx].reset()
                        ag.genome.macros[m_idx].start()
                        # immediately take the first step
                        a_type, prms, done = ag.genome.macros[m_idx].step(ag, world)
                        action_type = a_type; params = prms; ag.active_macro_done = done
                    else:
                        action_type = 0  # rest fallback

            # Execute primitive action
            cost = 0.0
            if action_type == 1:  # move
                cost += ag.act_move(params.get("move_v", np.zeros(2)), world)
            elif action_type == 2:  # harvest
                cost += ag.act_harvest(world)
            elif action_type == 3:  # transfer (share/steal)
                cost += ag.act_transfer(neighbors, params.get("transfer_delta", 0.0))
            elif action_type == 4:  # emit
                ch = int(params.get("emit_channel", 0)) % cfg.comm.channels
                r = float(params.get("emit_radius", 1.0))
                vec = params.get("emit_vec", np.zeros(cfg.comm.bandwidth, dtype=np.float32))
                cost += ag.act_emit(world, ch, vec, r)
            elif action_type == 5:  # mate (intent)
                cost += ag.act_mate(neighbors, world)
                self.intent_mate[ag.id] = self.world.tick  # mark intent by tick

            # Energy bookkeeping
            ag.energy -= cost
            # Clamp to [0, E_max]
            ag.energy = float(np.clip(ag.energy, 0.0, cfg.physics.E_max))

        # 3) Resolve sexual reproduction (if enabled)
        if cfg.reproduction.mode == "sexual":
            self.resolve_mating_events()

        # 4) Asexual reproduction (optional alongside sexual? Keep exclusive)
        if cfg.reproduction.mode == "asexual":
            self.resolve_asexual_births()

        # 5) Remove dead or out-of-range
        self.remove_dead()
        self.world.tick += 1

    def resolve_mating_events(self):
        """
        Pair agents that expressed mate intent within the recent window and are
        within mate_radius, using spatial buckets for O(k) local scans.
        Requires:
        - self._buckets   (built earlier in step())
        - self.intent_mate: Dict[int, int]  (agent.id -> last_intent_tick)
        - optional self.last_mated_tick: Dict[int, int]  (refractory)
        """
        if not getattr(self, "_buckets", None):  # no buckets built this tick
            return
        if not self.intent_mate:  # nobody expressed intent
            return

        cfg = self.cfg
        world = self.world
        H, W = cfg.world.grid_size
        torus = cfg.world.torus

        E_rep   = cfg.reproduction.E_rep
        frac    = cfg.reproduction.offspring_energy_frac
        radius  = cfg.reproduction.mate_radius
        window  = cfg.reproduction.mate_intent_window
        refr    = getattr(cfg.reproduction, "refractory_ticks", 0)
        tnow    = world.tick

        r_cells = int(math.ceil(radius))
        r2      = radius * radius

        used = set()
        new_agents: List[Agent] = []

        # Helper: torus-aware squared distance
        def d2(a_pos, b_pos):
            dx = abs(a_pos[0] - b_pos[0]); dy = abs(a_pos[1] - b_pos[1])
            if torus:
                dx = min(dx, H - dx); dy = min(dy, W - dy)
            return dx*dx + dy*dy

        # Iterate agents; try to find a single best partner per agent
        for i, ag in enumerate(self.agents):
            if (not ag.alive) or (ag.id in used):
                continue

            # intent recency & refractory checks for ag
            t0 = self.intent_mate.get(ag.id, -10**9)
            if tnow - t0 >= window:
                continue
            if refr and (tnow - getattr(self, "last_mated_tick", {}).get(ag.id, -10**9) < refr):
                continue

            # Bucket of ag
            x0 = int(ag.pos[0]) % H
            y0 = int(ag.pos[1]) % W

            best = None
            best_d2 = 1e18

            # Scan only buckets in the r_cells neighborhood
            for dx in range(-r_cells, r_cells + 1):
                xi = (x0 + dx) % H if torus else x0 + dx
                if not (0 <= xi < H):
                    continue
                row = self._buckets[xi]
                for dy in range(-r_cells, r_cells + 1):
                    yj = (y0 + dy) % W if torus else y0 + dy
                    if not (0 <= yj < W):
                        continue
                    for idx in row[yj]:
                        nb = self.agents[idx]
                        if (not nb.alive) or (nb.id == ag.id) or (nb.id in used):
                            continue
                        # partner intent recency & refractory
                        t1 = self.intent_mate.get(nb.id, -10**9)
                        if tnow - t1 >= window:
                            continue
                        if refr and (tnow - getattr(self, "last_mated_tick", {}).get(nb.id, -10**9) < refr):
                            continue
                        # distance test
                        dd = d2(ag.pos, nb.pos)
                        if dd <= r2 and dd < best_d2:
                            best = nb
                            best_d2 = dd

            if best is None:
                continue

            # Energy gate: both must have at least E_rep/2
            if ag.energy < E_rep/2 or best.energy < E_rep/2:
                continue

            # Pay reproduction cost
            ag.energy   -= E_rep/2
            best.energy -= E_rep/2
            E_offs = frac * E_rep

            # Child at midpoint; recombine genomes
            child_pos = (ag.pos + best.pos) / 2.0
            child_genome = recombine_genomes(ag.genome, best.genome, cfg)
            child = Agent(child_pos.copy(), E_offs, child_genome, cfg,
                        lineage_id=ag.lineage_id, parent_id=ag.id)

            # Bookkeeping
            ag.children_count += 1
            best.children_count += 1
            new_agents.append(child)

            used.add(ag.id); used.add(best.id)
            # Clear intents so they don't auto-remate next tick
            self.intent_mate.pop(ag.id, None)
            self.intent_mate.pop(best.id, None)

            # Refractory stamps
            if refr:
                if not hasattr(self, "last_mated_tick"):
                    self.last_mated_tick = {}
                self.last_mated_tick[ag.id] = tnow
                self.last_mated_tick[best.id] = tnow

        # Add all newborns after pairing loop
        if new_agents:
            self.agents.extend(new_agents)


    def resolve_asexual_births(self):
        E_rep = self.cfg.reproduction.E_rep
        frac = self.cfg.reproduction.offspring_energy_frac
        new_agents: List[Agent] = []
        for ag in self.agents:
            if not ag.alive:
                continue
            if ag.energy >= E_rep:
                ag.energy -= E_rep
                E_offs = frac * E_rep
                child_pos = ag.pos + np.random.uniform(-0.5, 0.5, size=2)
                child_pos = self.world.wrap_pos(child_pos)
                child_genome = ag.genome.clone_mutated()
                child = Agent(child_pos.copy(), E_offs, child_genome, self.cfg, lineage_id=ag.lineage_id, parent_id=ag.id)
                ag.children_count += 1
                new_agents.append(child)
        self.agents.extend(new_agents)

# ---------------------------
# Recombination (sexual)
# ---------------------------

def recombine_genomes(g1: Genome, g2: Genome, cfg: Config) -> Genome:
    # average capacities + noise
    child = Genome(
        sensor_radius=max(1.0, (g1.sensor_radius + g2.sensor_radius)/2 + np.random.normal(0, 0.1)),
        movement_eff=max(0.6, (g1.movement_eff + g2.movement_eff)/2 + np.random.normal(0,0.02)),
        comm_bandwidth=max(1, int(round((g1.comm_bandwidth + g2.comm_bandwidth)/2 + np.random.normal(0,0.2)))),
        comm_radius_max=max(1.0, (g1.comm_radius_max + g2.comm_radius_max)/2 + np.random.normal(0,0.1)),
        controller=None,
        macros=[],
        mut_sigma=max(0.01, (g1.mut_sigma + g2.mut_sigma)/2 + np.random.normal(0,0.005)),
        macro_param_sigma=max(0.02, (g1.macro_param_sigma + g2.macro_param_sigma)/2 + np.random.normal(0,0.01))
    )
    # controller: average weights + small noise
    ctrl = g1.controller.clone()
    for (arr, arr1, arr2) in [
        (ctrl.Wxh, g1.controller.Wxh, g2.controller.Wxh),
        (ctrl.Whh, g1.controller.Whh, g2.controller.Whh),
        (ctrl.bh,  g1.controller.bh,  g2.controller.bh ),
        (ctrl.Who, g1.controller.Who, g2.controller.Who),
        (ctrl.bo,  g1.controller.bo,  g2.controller.bo ),
        (ctrl.Wparam, g1.controller.Wparam, g2.controller.Wparam),
        (ctrl.bparam, g1.controller.bparam, g2.controller.bparam),
        (ctrl.Wchan, g1.controller.Wchan, g2.controller.Wchan),
        (ctrl.bchan, g1.controller.bchan, g2.controller.bchan),
    ]:
        arr[:] = 0.5*(arr1 + arr2) + np.random.normal(0, cfg.evolution.mutation_sigma, size=arr.shape)
    child.controller = ctrl
    # macros: randomly inherit from one parent, mutate lightly
    base_macros = (g1.macros if np.random.rand() < 0.5 else g2.macros)
    child.macros = [Macro(steps=[MacroStep(**vars(s)) for s in m.steps]) for m in base_macros]
    # small mutate
    child = child.clone_mutated()
    return child

# ---------------------------
# Initialization helpers
# ---------------------------

def init_population(cfg: Config, world: World) -> Population:
    pop = Population(cfg, world)
    H, W = cfg.world.grid_size
    for i in range(cfg.evolution.pop_size):
        pos = np.array([np.random.uniform(0, H-1), np.random.uniform(0, W-1)], dtype=np.float32)
        energy = np.random.uniform(cfg.physics.E_max*0.4, cfg.physics.E_max*0.8)
        # Controller action space is 6 + max_macros (we'll allow up to 4 macros)
        action_space = 6 + 4
        ctrl = TinyRNNController(
            obs_size=max(16, 8 + cfg.comm.channels*cfg.comm.bandwidth),
            hidden_size=cfg.controller.hidden_size,
            action_space=action_space,
            channels=cfg.comm.channels,
            bandwidth=cfg.comm.bandwidth,
            init_scale=cfg.controller.init_weight_scale
        )
        macros = []
        for _ in range(cfg.macros.init_count):
            macros.append(random_macro(cfg.comm.bandwidth))
        genome = Genome(
            sensor_radius=2.0 + np.random.uniform(-0.5, 0.5),
            movement_eff=1.0 + np.random.uniform(-0.1, 0.1),
            comm_bandwidth=cfg.comm.bandwidth,
            comm_radius_max=cfg.comm.r_max,
            controller=ctrl,
            macros=macros,
            mut_sigma=cfg.evolution.mutation_sigma,
            macro_param_sigma=cfg.macros.param_sigma
        )
        ag = Agent(pos, energy, genome, cfg)
        pop.add_agent(ag)
    return pop

# ---------------------------
# Logging
# ---------------------------

@dataclass
class LogState:
    t: int
    pop_size: int
    mean_energy: float
    births: int
    deaths: int
    messages: float
    harvest: float
    transfers: float

class Logger:
    def __init__(self):
        self.history: List[LogState] = []
        self.last_alive_ids: set = set()
        self.last_message_sum: float = 0.0
        self.last_harvest_sum: float = 0.0
        self.last_transfers_sum: float = 0.0  # proxy via energy deltas from transfers (not exact)

    def snapshot(self, t: int, pop: Population, world: World):
        alive_ids = {ag.id for ag in pop.agents if ag.alive}
        births = max(0, len(alive_ids) - len(self.last_alive_ids))
        deaths = max(0, len(self.last_alive_ids) - len(alive_ids))
        self.last_alive_ids = alive_ids
        # message magnitude proxy
        msg_sum = float(np.sum(np.abs(world.message_field)))
        msg_delta = msg_sum - self.last_message_sum
        self.last_message_sum = msg_sum
        # harvest proxy
        harv = float(np.sum(world.H_t))  # note: H_t resets each step; snapshot before reset would be better
        self.last_harvest_sum = harv
        # transfers proxy is not tracked precisely here; placeholder 0
        self.history.append(LogState(
            t=t,
            pop_size=len(alive_ids),
            mean_energy=float(np.mean([ag.energy for ag in pop.agents])) if pop.agents else 0.0,
            births=births,
            deaths=deaths,
            messages=msg_delta,
            harvest=harv,
            transfers=0.0
        ))

    def to_dict(self) -> Dict[str, List[Dict[str, Any]]]:
        return {"history": [vars(h) for h in self.history]}

# ---------------------------
# Main demo
# ---------------------------

def run_demo():
    cfg = Config()
    np.random.seed(cfg.run.rng_seed)
    random.seed(cfg.run.rng_seed)

    world = World(cfg)
    pop = init_population(cfg, world)
    logger = Logger()

    T = cfg.run.ticks
    for t in range(T):
        pop.step()
        if (t % cfg.run.log_interval) == 0:
            logger.snapshot(t, pop, world)
            print(f"[t={t}] pop={len(pop.agents)} meanE={np.mean([ag.energy for ag in pop.agents]):.2f}")

        if len(pop.agents) == 0:
            print(f"All agents died at t={t}.")
            break
    # final snapshot
    logger.snapshot(T, pop, world)
    # dump a small JSON summary
    summary_path = "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(logger.to_dict(), f, indent=2)

    print(f"Saved summary to {summary_path}.")

if __name__ == "__main__":
    run_demo()
