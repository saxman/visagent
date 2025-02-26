import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Ensure we're using GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")


class SEIRModel:
    def __init__(self, population, beta, sigma, gamma, initial_conditions):
        """
        Initializes the SEIR model.
        :param population: Total population (N)
        :param beta: Transmission rate (beta)
        :param sigma: Rate at which exposed individuals become infectious (1/incubation period)
        :param gamma: Recovery rate (1/infectious period)
        :param initial_conditions: Tuple (S0, E0, I0, R0) of initial conditions
        """
        self.N = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.S0, self.E0, self.I0, self.R0 = initial_conditions

    def deriv(self, y, t):
        """Defines the differential equations for the SEIR model."""
        S, E, I, R = y
        dSdt = -self.beta * S * I / self.N
        dEdt = self.beta * S * I / self.N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return dSdt, dEdt, dIdt, dRdt

    def run_simulation(self, days):
        """
        Runs the SEIR simulation.
        :param days: Number of days to simulate
        :return: Tuple of arrays (S, E, I, R)
        """
        t = np.linspace(0, days, days)
        y0 = (self.S0, self.E0, self.I0, self.R0)
        result = odeint(self.deriv, y0, t)
        S, E, I, R = result.T
        return t, S, E, I, R

    def plot(self, t, S, E, I, R):
        """
        Plots the SEIR model results.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, "b", label="Susceptible")
        plt.plot(t, E, "y", label="Exposed")
        plt.plot(t, I, "r", label="Infectious")
        plt.plot(t, R, "g", label="Recovered")

        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title("SEIR Model Simulation")
        plt.legend()
        plt.grid()
        plt.show()


class SEIRTorchModel(nn.Module):
    def __init__(self, population, beta, sigma, gamma, initial_conditions):
        """
        Initializes the SEIR model.
        :param population: Total population (N)
        :param beta: Transmission rate (beta)
        :param sigma: Rate at which exposed individuals become infectious (1/incubation period)
        :param gamma: Recovery rate (1/infectious period)
        :param initial_conditions: Tuple (S0, E0, I0, R0) of initial conditions
        """
        super(SEIRTorchModel, self).__init__()
        self.N = population
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.S0, self.E0, self.I0, self.R0 = initial_conditions

    def deriv(self, y):
        """Defines the differential equations for the SEIR model."""
        S, E, I, R = y
        dSdt = -self.beta * S * I / self.N
        dEdt = self.beta * S * I / self.N - self.sigma * E
        dIdt = self.sigma * E - self.gamma * I
        dRdt = self.gamma * I
        return torch.tensor([dSdt, dEdt, dIdt, dRdt], dtype=torch.float32).to(device)

    def run_simulation(self, days, dt=1.0):
        """
        Runs the SEIR simulation.
        :param days: Number of days to simulate
        :return: Tuple of tensors (S, E, I, R)
        """
        t = torch.arange(0, days, dt, dtype=torch.float32).to(device)
        y = torch.tensor([self.S0, self.E0, self.I0, self.R0], dtype=torch.float32).to(device)
        results = [y.clone()]

        for _ in t[1:]:
            y = y + self.deriv(y) * dt
            results.append(y.clone())

        results = torch.stack(results).cpu().numpy()
        return t.cpu().numpy(), results[:, 0], results[:, 1], results[:, 2], results[:, 3]

    def plot(self, t, S, E, I, R):
        """
        Plots the SEIR model results.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(t, S, "b", label="Susceptible")
        plt.plot(t, E, "y", label="Exposed")
        plt.plot(t, I, "r", label="Infectious")
        plt.plot(t, R, "g", label="Recovered")

        plt.xlabel("Days")
        plt.ylabel("Population")
        plt.title("SEIR Model Simulation with PyTorch")
        plt.legend()
        plt.grid()
        plt.show()
