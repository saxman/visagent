from visagent.agent_based import SEIRModel, SEIRTorchModel, SEIRTorchParallelModel
import time

population = 10000  # 10K
beta = 0.3  # Transmission rate
sigma = 1 / 5.2  # Incubation period (average 5.2 days)
gamma = 1 / 7  # Recovery rate (average 7 days infectious period)
initial_conditions = (int(population * 0.99), int(population * 0.01), 0, 0)  # Initial (S, E, I, R)
space_size = 100
infection_radius = 1
days = 160


def run_standard_model():
    start = time.perf_counter()
    model = SEIRModel(population, beta, sigma, gamma, initial_conditions, space_size, infection_radius)
    results = model.run_simulation(days)
    end = time.perf_counter()

    print(f"standard model: {end - start:.6f} seconds")
    model.plot(results)


def run_torch_model():
    start = time.perf_counter()
    model = SEIRTorchModel(population, beta, sigma, gamma, initial_conditions, space_size, infection_radius)
    results = model.run_simulation(days)
    end = time.perf_counter()

    print(f"torch model: {end - start:.6f} seconds")
    model.plot(results)


def run_torch_parallel_model():
    start = time.perf_counter()
    model = SEIRTorchParallelModel(population, beta, sigma, gamma, initial_conditions, space_size, infection_radius)
    results = model.run_simulation(days)
    end = time.perf_counter()

    print(f"torch parallel model: {end - start:.6f} seconds")
    model.plot(results)


if __name__ == "__main__":
    # run_standard_model()
    # run_torch_model()
    run_torch_parallel_model()
