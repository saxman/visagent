import torch
import numpy as np

# Ensure we're using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")
# device = torch.device("cpu")


import random
import matplotlib.pyplot as plt


class SEIRAgent:
    def __init__(self, state):
        self.state = state
        self.days_in_state = 0

    def update(self, beta, sigma, gamma, agents):
        if self.state == "S":
            infected_neighbors = sum(1 for agent in agents if agent.state == "I")
            if random.random() < 1 - (1 - beta) ** infected_neighbors:
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


class SEIRModel:
    def __init__(self, population, beta, sigma, gamma, initial_conditions):
        self.population = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.agents = (
            [SEIRAgent("S") for _ in range(initial_conditions[0])]
            + [SEIRAgent("E") for _ in range(initial_conditions[1])]
            + [SEIRAgent("I") for _ in range(initial_conditions[2])]
            + [SEIRAgent("R") for _ in range(initial_conditions[3])]
        )
        random.shuffle(self.agents)

    def run_simulation(self, days):
        results = {"S": [], "E": [], "I": [], "R": []}
        for _ in range(days):
            for agent in self.agents:
                agent.update(self.beta, self.sigma, self.gamma, self.agents)
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
