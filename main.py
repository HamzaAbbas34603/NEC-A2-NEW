from src.dataset_loader import parse_dataset
from src.genetic_algorithm import genetic_algorithm
from src.utils import plot_fitness

if __name__ == "__main__":
    dataset_path = "dataset/jobshop1.txt"
    jobs = parse_dataset(dataset_path)

    # Run Genetic Algorithm with different parameters
    solution, fitness, fitness_history = genetic_algorithm(jobs, population_size=50, generations=200)

    print("Best Solution:", solution)
    print("Best Fitness (Makespan):", fitness)
    plot_fitness(fitness_history, title="Genetic Algorithm - Fitness Evolution")
