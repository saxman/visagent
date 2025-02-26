from visagent.agent_based import SEIRModel
import time

population = 10000  # 10K
beta = 0.3  # Transmission rate
sigma = 1 / 5.2  # Incubation period (average 5.2 days)
gamma = 1 / 7  # Recovery rate (average 7 days infectious period)
initial_conditions = (int(population * 0.99), int(population * 0.01), 0, 0)  # Initial (S, E, I, R)
days = 160

# Example usage:
if __name__ == "__main__":
    model = SEIRModel(population, beta, sigma, gamma, initial_conditions)
    results = model.run_simulation(days)
    model.plot(results)
