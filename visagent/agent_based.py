import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from visagent import utils


class SEIRAgent:
    def __init__(self, state, position):
        self.state = state
        self.position = position  # (x, y) coordinate in 2D space
        self.days_in_state = 0

    def update(self, beta, sigma, gamma, agents, distance_matrix, infection_radius):
        if self.state == "S":
            for i, agent in enumerate(agents):
                if agent.state == "I" and distance_matrix[self.position][agent.position] <= infection_radius:
                    if random.random() < beta:
                        self.state = "E"
                        break
        elif self.state == "E":
            self.days_in_state += 1
            if self.days_in_state >= int(1 / sigma):
                self.state = "I"
                self.days_in_state = 0
        elif self.state == "I":
            self.days_in_state += 1
            if self.days_in_state >= int(1 / gamma):
                self.state = "R"
                self.days_in_state = 0


class SEIRModel:
    def __init__(self, population, beta, sigma, gamma, initial_conditions, space_size, infection_radius):
        self.population = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.infection_radius = infection_radius
        self.positions = [(random.uniform(0, space_size), random.uniform(0, space_size)) for _ in range(population)]
        self.agents = (
            [SEIRAgent("S", self.positions[i]) for i in range(initial_conditions[0])]
            + [
                SEIRAgent("E", self.positions[i])
                for i in range(initial_conditions[0], initial_conditions[0] + initial_conditions[1])
            ]
            + [
                SEIRAgent("I", self.positions[i])
                for i in range(
                    initial_conditions[0] + initial_conditions[1],
                    initial_conditions[0] + initial_conditions[1] + initial_conditions[2],
                )
            ]
            + [
                SEIRAgent("R", self.positions[i])
                for i in range(initial_conditions[0] + initial_conditions[1] + initial_conditions[2], population)
            ]
        )
        random.shuffle(self.agents)
        self.distance_matrix = self.compute_distance_matrix()

    def compute_distance_matrix(self):
        distance_matrix = {}
        for i, pos1 in enumerate(self.positions):
            distance_matrix[pos1] = {}
            for j, pos2 in enumerate(self.positions):
                distance_matrix[pos1][pos2] = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return distance_matrix

    def run_simulation(self, days):
        results = {"S": [], "E": [], "I": [], "R": []}
        for _ in tqdm(range(days)):
            for agent in self.agents:
                agent.update(
                    self.beta, self.sigma, self.gamma, self.agents, self.distance_matrix, self.infection_radius
                )
            results["S"].append(sum(1 for agent in self.agents if agent.state == "S"))
            results["E"].append(sum(1 for agent in self.agents if agent.state == "E"))
            results["I"].append(sum(1 for agent in self.agents if agent.state == "I"))
            results["R"].append(sum(1 for agent in self.agents if agent.state == "R"))
        return results

    def plot(self, results):
        plt.figure(figsize=(10, 6))
        plt.plot(results["S"], "b", label="Susceptible")
        plt.plot(results["E"], "y", label="Exposed")
        plt.plot(results["I"], "r", label="Infectious")
        plt.plot(results["R"], "g", label="Recovered")

        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title("Agent-Based SEIR Model Simulation")
        plt.legend()
        plt.grid()
        plt.show()


class SEIRTorchAgent:
    def __init__(self, state, index, device):
        self.state = state
        self.index = index
        self.days_in_state = 0

        self.device = device

    def update(self, beta, sigma, gamma, infected_indices, distance_matrix, infection_radius):
        if self.state == "S":
            distances = distance_matrix[self.index, infected_indices]
            if torch.any(distances <= infection_radius):
                if random.random() < beta:
                    self.state = "E"
        elif self.state == "E":
            self.days_in_state += 1
            if self.days_in_state >= int(1 / sigma):
                self.state = "I"
                self.days_in_state = 0
        elif self.state == "I":
            self.days_in_state += 1
            if self.days_in_state >= int(1 / gamma):
                self.state = "R"
                self.days_in_state = 0


class SEIRTorchModel:
    def __init__(self, population, beta, sigma, gamma, initial_conditions, space_size, infection_radius):
        self.device = utils.get_pytorch_device()

        self.population = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.infection_radius = infection_radius
        self.positions = torch.rand((population, 2), dtype=torch.float32, device=self.device) * space_size
        self.agents = (
            [SEIRTorchAgent("S", i) for i in range(initial_conditions[0])]
            + [
                SEIRTorchAgent("E", i)
                for i in range(initial_conditions[0], initial_conditions[0] + initial_conditions[1])
            ]
            + [
                SEIRTorchAgent("I", i)
                for i in range(
                    initial_conditions[0] + initial_conditions[1],
                    initial_conditions[0] + initial_conditions[1] + initial_conditions[2],
                )
            ]
            + [
                SEIRTorchAgent("R", i)
                for i in range(initial_conditions[0] + initial_conditions[1] + initial_conditions[2], population)
            ]
        )
        random.shuffle(self.agents)
        self.distance_matrix = self.compute_distance_matrix()

    def compute_distance_matrix(self):
        positions = self.positions.to(self.device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance_matrix = torch.norm(diff, dim=2)
        return distance_matrix

    def run_simulation(self, days):
        results = {"S": [], "E": [], "I": [], "R": []}
        for _ in tqdm(range(days)):
            infected_indices = torch.tensor(
                [agent.index for agent in self.agents if agent.state == "I"], dtype=torch.long, device=self.device
            )
            for agent in self.agents:
                agent.update(
                    self.beta, self.sigma, self.gamma, infected_indices, self.distance_matrix, self.infection_radius
                )
            results["S"].append(sum(1 for agent in self.agents if agent.state == "S"))
            results["E"].append(sum(1 for agent in self.agents if agent.state == "E"))
            results["I"].append(sum(1 for agent in self.agents if agent.state == "I"))
            results["R"].append(sum(1 for agent in self.agents if agent.state == "R"))
        return results

    def plot(self, results):
        plt.figure(figsize=(10, 6))
        plt.plot(results["S"], "b", label="Susceptible")
        plt.plot(results["E"], "y", label="Exposed")
        plt.plot(results["I"], "r", label="Infectious")
        plt.plot(results["R"], "g", label="Recovered")

        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title("Agent-Based SEIR Model Simulation with Distance Matrix (GPU Accelerated)")
        plt.legend()
        plt.grid()
        plt.show()


class SEIRTorchParallelModel:
    def __init__(self, population, beta, sigma, gamma, initial_conditions, space_size, infection_radius):
        self.device = utils.get_pytorch_device()

        self.population = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.infection_radius = infection_radius

        # Randomly assign agent positions in a 2D space
        self.positions = torch.rand((population, 2), dtype=torch.float32, device=self.device) * space_size

        # Initialize states (0: S, 1: E, 2: I, 3: R)
        self.states = torch.zeros((population,), dtype=torch.int, device=self.device)
        self.states[: initial_conditions[0]] = 0  # Susceptible
        self.states[initial_conditions[0] : initial_conditions[0] + initial_conditions[1]] = 1  # Exposed
        self.states[
            initial_conditions[0] + initial_conditions[1] : initial_conditions[0]
            + initial_conditions[1]
            + initial_conditions[2]
        ] = 2  # Infected
        self.states[initial_conditions[0] + initial_conditions[1] + initial_conditions[2] :] = 3  # Recovered

        # Track number of days an agent has been in a state
        self.days_in_state = torch.zeros((population,), dtype=torch.int, device=self.device)

        # Precompute the distance matrix
        self.distance_matrix = self.compute_distance_matrix()

    def compute_distance_matrix(self):
        """Compute pairwise Euclidean distances between agents."""
        positions = self.positions.to(self.device)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        distance_matrix = torch.norm(diff, dim=2)
        return distance_matrix

    def run_simulation(self, days):
        """Run the SEIR model simulation for a given number of days."""
        results = {"S": [], "E": [], "I": [], "R": []}

        for _ in range(days):
            # Identify infected and susceptible individuals
            infected_indices = (self.states == 2).nonzero(as_tuple=True)[0]
            susceptible_indices = (self.states == 0).nonzero(as_tuple=True)[0]

            # Compute distances between susceptible and infected individuals
            distances = self.distance_matrix[susceptible_indices][:, infected_indices]
            infections = (distances <= self.infection_radius).any(dim=1)
            infection_mask = torch.rand_like(infections.float(), device=self.device) < self.beta

            # Update newly exposed individuals
            self.states[susceptible_indices[infections & infection_mask]] = 1

            # Increment time spent in current state
            self.days_in_state += 1

            # Transition exposed individuals to infected
            exposed_mask = (self.states == 1) & (self.days_in_state >= int(1 / self.sigma))
            self.states[exposed_mask] = 2
            self.days_in_state[exposed_mask] = 0

            # Transition infected individuals to recovered
            infected_mask = (self.states == 2) & (self.days_in_state >= int(1 / self.gamma))
            self.states[infected_mask] = 3
            self.days_in_state[infected_mask] = 0

            # Store daily results
            results["S"].append((self.states == 0).sum().item())
            results["E"].append((self.states == 1).sum().item())
            results["I"].append((self.states == 2).sum().item())
            results["R"].append((self.states == 3).sum().item())

        return results

    def plot(self, results):
        plt.figure(figsize=(10, 6))
        plt.plot(results["S"], "b", label="Susceptible")
        plt.plot(results["E"], "y", label="Exposed")
        plt.plot(results["I"], "r", label="Infectious")
        plt.plot(results["R"], "g", label="Recovered")

        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title("Agent-Based SEIR Model Simulation with Distance Matrix (GPU Accelerated)")
        plt.legend()
        plt.grid()
        plt.show()
