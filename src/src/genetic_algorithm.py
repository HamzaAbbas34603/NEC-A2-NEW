
import random
from .ga_components import (
    generate_chromosome, calculate_makespan, tournament_selection,
    roulette_wheel_selection, order_crossover, cycle_crossover,
    mutate, scramble_mutation
)

def genetic_algorithm(jobs, population_size=50, generations=50, crossover_rate=0.8, mutation_rate=0.2, 
                      selection_method="tournament", crossover_method="order", mutation_method="swap"):
    population = [generate_chromosome(jobs) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    for generation in range(generations):
        # Evaluate fitness
        fitnesses = [calculate_makespan(chrom, jobs) for chrom in population]
        min_idx = fitnesses.index(min(fitnesses))
        if fitnesses[min_idx] < best_fitness:
            best_fitness = fitnesses[min_idx]
            best_solution = population[min_idx]
        fitness_history.append(best_fitness)

        # Selection
        new_population = []
        for _ in range(population_size // 2):
            if selection_method == "tournament":
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
            elif selection_method == "roulette":
                parent1 = roulette_wheel_selection(population, fitnesses)
                parent2 = roulette_wheel_selection(population, fitnesses)
            
            # Crossover
            if random.random() < crossover_rate:
                if crossover_method == "order":
                    child1 = order_crossover(parent1, parent2)
                    child2 = order_crossover(parent2, parent1)
                elif crossover_method == "cycle":
                    child1 = cycle_crossover(parent1, parent2)
                    child2 = cycle_crossover(parent2, parent1)
            else:
                child1, child2 = parent1, parent2
            
            new_population.extend([child1, child2])

        # Mutation
        for individual in new_population:
            if random.random() < mutation_rate:
                if mutation_method == "swap":
                    mutate(individual)
                elif mutation_method == "scramble":
                    scramble_mutation(individual)

        # Replace population
        population = new_population

    return best_solution, best_fitness, fitness_history
