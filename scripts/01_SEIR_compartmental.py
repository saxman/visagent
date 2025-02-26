from visagent.compartmental import SEIRModel, SEIRTorchModel
import time

population = 10000000  # 10 million
beta = 0.3  # Transmission rate
sigma = 1 / 5.2  # Incubation period (average 5.2 days)
gamma = 1 / 7  # Recovery rate (average 7 days infectious period)
initial_conditions = (population * 0.99, population * 0.01, 0, 0)  # Initial (S, E, I, R)
days = 160


def run_standard_model():
    start = time.perf_counter()
    model = SEIRTorchModel(population, beta, sigma, gamma, initial_conditions)
    t, S, E, I, R = model.run_simulation(160)
    end = time.perf_counter()

    print(f"standard model: {end - start:.6f} seconds")
    # model.plot(160)


def run_torch_model():
    start = time.perf_counter()
    model = SEIRTorchModel(population, beta, sigma, gamma, initial_conditions)
    t, S, E, I, R = model.run_simulation(160)
    end = time.perf_counter()

    print(f"torch model: {end - start:.6f} seconds")
    # model.plot(t, S, E, I, R)


if __name__ == "__main__":
    run_standard_model()
    run_torch_model()
