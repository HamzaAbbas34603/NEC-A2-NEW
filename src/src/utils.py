# Utility functions (e.g., visualization)
import matplotlib.pyplot as plt

def plot_fitness(fitness_history, title="Fitness Evolution"):
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history, label="Best Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness (Makespan)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()
