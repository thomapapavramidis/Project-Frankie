# === evolutionary_agents_sim.py ===

import numpy as np
import random
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import pandas as pd

# Global config
GRID_SIZE = 10
NUM_AGENTS = 30
NUM_GENERATIONS = 1000
STEPS_PER_AGENT = 200
MUTATION_RATE = 0.05

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

# === Utilities ===
def random_position():
    return (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))

def mutate_weights(weights, rate=MUTATION_RATE):
    return weights + np.random.randn(*weights.shape) * rate

# === Environment ===
class Environment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE))
        self.food_positions = set()
        self.water_positions = set()
        self.hazard_positions = set()
        self.safe_zones = set()
        self.spawn_resources()

    def spawn_resources(self):
        self.food_positions = set()
        self.water_positions = set()
        self.hazard_positions = ({random_position() for _ in range(random.randint(5, 15))} | {random_position() for _ in range(random.randint(3, 10))})
        self.safe_zones = {random_position() for _ in range(random.randint(3, 10))}


    def get_tile_content(self, pos):
        return {
            "food": pos in self.food_positions,
            "water": pos in self.water_positions,
            "hazard": pos in self.hazard_positions,
            "safe": pos in self.safe_zones
        }
    
    def update_hazards(self):
        self.hazard_positions = {random_position() for _ in range(random.randint(5, 10))}


# === Agent ===
class Agent:
    def __init__(self, parent=None, weights=None):
        self.memory_vector = np.zeros(6)
        if parent:
            self.policy_weights = mutate_weights(parent.policy_weights)
            self.last_message = parent.last_message[:]
            self.messages_heard = parent.messages_heard[-10:]  # pass cultural memory
            self.curiosity = max(0.0, min(1.0, parent.curiosity + np.random.uniform(-0.05, 0.05)))
            self.trust = max(0.0, min(1.0, getattr(parent, 'trust', 0.5) + np.random.uniform(-0.05, 0.05)))
            self.aggression = max(0.0, min(1.0, getattr(parent, 'aggression', 0.5) + np.random.uniform(-0.05, 0.05)))
        else:
            self.policy_weights = weights if weights is not None else np.random.randn(12, 8)
            self.last_message = []
            self.messages_heard = []
            
            self.curiosity = 1.0
            self.trust = 0.5
            self.aggression = 0.5

        self.pos = random_position()
        self.hunger = 1.0
        self.thirst = 1.0
        self.fatigue = 0.0
        self.pain = 0.0
        self.reproduction_urge = 0.0
        self.curiosity = 1.0
        self.age = 0
        self.alive = True
        self.inventory = {"food": 0, "water": 0}

        self.speech_reward = 0.0  # reward is not currently used but tracked

    def perceive(self):
        return np.concatenate((
            np.array([
                self.hunger,
                self.thirst,
                self.fatigue,
                self.pain,
                self.reproduction_urge,
                self.curiosity
            ]),
            self.memory_vector
        ))

    def act(self):
        input_vec = self.perceive()  # now includes memory_vector
        logits = input_vec @ self.policy_weights
        action = np.argmax(logits)
        return action

    def generate_speech(self):
        # === GPT-2 version (commented out for performance) ===
        # state = self.perceive()
        # prompt = f"You are a survival agent. Hunger={state[0]:.2f}, Thirst={state[1]:.2f}, Fatigue={state[2]:.2f}, Pain={state[3]:.2f}, Reproduction={state[4]:.2f}, Curiosity={state[5]:.2f}. Say how you feel:"
        # inputs = tokenizer(prompt, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=12,
        #         do_sample=True,
        #         top_k=50,
        #         pad_token_id=tokenizer.eos_token_id
        #     )
        # decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # message = decoded.replace(prompt, "").strip()
        # message = ''.join(c for c in message if c.isalnum() or c.isspace())
        # message = message.split()[:8]  # limit to 8 words

        # === Fallback to fast random-word speech ===
        vocab = ["food", "water", "help", "pain", "safe", "danger", "thanks", "share", "thirst", "hungry"]
        message = random.choices(vocab, k=3)
        self.last_message = message
        return message

    def interpret_message(self, message):
        self.messages_heard.append(message)
        input_text = "Agent heard: " + " ".join(message) + ". Meaning:"
        inputs = tokenizer(input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id  # silences warning
            )
        meaning = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(input_text, "").strip()
        print(f"Interpreted message: '{' '.join(message)}' → '{meaning}'")


    def step(self, env, others):
        if not self.alive:
            return

        action = self.act()
        x, y = self.pos

        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < GRID_SIZE - 1: y += 1
        elif action == 2 and x < GRID_SIZE - 1: x += 1
        elif action == 3 and x > 0: x -= 1
        elif action == 4: pass
        elif action == 5:
            if self.inventory["food"] > 0:
                self.hunger = min(1.0, self.hunger + 0.5)
                self.inventory["food"] -= 1
        elif action == 6:
            for other in others:
                if other.pos == self.pos and other != self and self.inventory["food"] > 0:
                    other.inventory["food"] += 1
                    self.inventory["food"] -= 1
        elif action == 7 and self.age % 10 == 0:
            message = self.generate_speech()
            #for other in others:
               # if other.pos == self.pos and other != self:
                #    other.interpret_message(message)

        self.pos = (x, y)
        content = env.get_tile_content(self.pos)
        if content["food"] and self.inventory["food"] < 4:
            self.inventory["food"] += 1
        if content["water"] and self.inventory["water"] < 4:
            self.inventory["water"] += 1
        if content["hazard"]:
            self.pain += 0.05
        if content["safe"]:
            self.fatigue = max(0.0, self.fatigue - 0.05)

        self.hunger -= 0.01
        self.thirst -= 0.015
        self.fatigue += 0.02 if (self.age // 20) % 2 == 1 else 0.01  # Night = more fatigue
        self.reproduction_urge += 0.005
        self.curiosity -= 0.002
        if len(self.messages_heard) > 0:
            self.curiosity += 0.001  # small boost if engaged in social interaction
        self.age += 1

        if self.hunger <= 0 or self.thirst <= 0 or self.fatigue >= 1.0 or self.pain >= 1.0 or self.age > 130:
            self.alive = False

# === Evolutionary Simulation ===
def simulate_generation(generation, agents):
    env = Environment()
    if generation == 0:
        agents = [Agent() for _ in range(NUM_AGENTS)]
        for i, a in enumerate(agents):
            a.inventory['food'] = 1
            a.inventory['water'] = 1
            a.has_reproduced = False
    else:
        for a in agents:
            if not hasattr(a, 'has_reproduced'):
                a.has_reproduced = False

    new_agents = []
    for step in range(STEPS_PER_AGENT):
        if step % 20 == 0:
            env.update_hazards()
        for agent in agents:
            if agent.alive:
                agent.step(env, agents)
                agent.memory_vector = 0.9 * agent.memory_vector + 0.1 * agent.perceive()[:6]
                if agent.reproduction_urge > 1.0 and not agent.has_reproduced:
                    child = Agent(parent=agent)
                    new_agents.append(child)
                    agent.has_reproduced = True

    import collections
    alive = sum(1 for a in agents if a.alive)
    if len(agents) == 0:
        print(f"Gen {generation}: All agents died. Respawning with mutations...")
        agents = [Agent() for _ in range(NUM_AGENTS)]
        for a in agents:
            a.inventory['food'] = 1
            a.inventory['water'] = 1
            a.has_reproduced = False
        return agents

    avg_age = sum(a.age for a in agents) / len(agents)
    avg_food = sum(a.inventory['food'] for a in agents) / len(agents)
    avg_water = sum(a.inventory['water'] for a in agents) / len(agents)
    msg_count = collections.Counter(tuple(msg) for a in agents for msg in a.messages_heard)
    print(f"Gen {generation}: {alive} alive, avg age {avg_age:.2f}, avg food {avg_food:.2f}, avg water {avg_water:.2f}")
    print(f"Most common messages: {msg_count.most_common(3)}")
    action_counts = collections.Counter([a.act() for a in agents if a.alive])
    print(f"Action usage: {dict(action_counts)}")
    avg_trust = sum(a.trust for a in agents) / len(agents)
    avg_aggression = sum(a.aggression for a in agents) / len(agents)
    print(f"Avg trust: {avg_trust:.2f}, Avg aggression: {avg_aggression:.2f}")

    output_data.append({
        'generation': generation,
        'alive': alive,
        'avg_age': avg_age,
        'avg_food': avg_food,
        'avg_water': avg_water,
        'avg_trust': avg_trust,
        'avg_aggression': avg_aggression
    })

    return new_agents

# === Run Simulation ===
output_data = []
agents = [Agent() for _ in range(NUM_AGENTS)]

new_agents = []

for generation in range(NUM_GENERATIONS):
    print(f"--- Generation {generation} ---")
    new_agents = simulate_generation(generation, agents)
    agents = [a for a in agents if a.alive] + new_agents
    if len(agents) > NUM_AGENTS:
        agents.sort(key=lambda a: (a.alive, a.age), reverse=True)
        survivors = agents[:5]  # pick top survivors
        mutants = [Agent(parent=random.choice(survivors)) for _ in range(NUM_AGENTS - len(survivors))]
        agents = survivors + mutants
    if generation % 100 == 0:
        pd.DataFrame(output_data).to_excel("evolution_output.xlsx", index=False)
