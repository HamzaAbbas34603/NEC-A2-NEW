# Genetic Algorithm components
import random

def generate_chromosome(jobs):
    chromosome = [(job_id, op_id) for job_id, ops in enumerate(jobs) for op_id in range(len(ops))]
    random.shuffle(chromosome)
    return chromosome

def calculate_makespan(chromosome, jobs):
    machine_times = {}
    job_times = {}
    for job_id, op_id in chromosome:
        machine, time = jobs[job_id][op_id]
        start_time = max(machine_times.get(machine, 0), job_times.get(job_id, 0))
        end_time = start_time + time
        machine_times[machine] = end_time
        job_times[job_id] = end_time
    return max(machine_times.values())

def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(range(len(population)), k)
    best = min(selected, key=lambda idx: fitnesses[idx])
    return population[best]

def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    pointer = end
    for gene in parent2:
        if gene not in child:
            if pointer == size:
                pointer = 0
            child[pointer] = gene
            pointer += 1
    return child

def mutate(chromosome):
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]

def roulette_wheel_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    probabilities = [1 - (f / total_fitness) for f in fitnesses]  # Higher fitness means higher probability
    cumulative_probs = [sum(probabilities[:i+1]) for i in range(len(probabilities))]
    rand = random.random()
    for idx, cumulative_prob in enumerate(cumulative_probs):
        if rand < cumulative_prob:
            return population[idx]

def cycle_crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    cycle_indices = []
    idx = 0
    while parent1[idx] not in cycle_indices:
        cycle_indices.append(parent1[idx])
        idx = parent1.index(parent2[idx])
    for i in range(size):
        if parent1[i] in cycle_indices:
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    return child

def scramble_mutation(chromosome):
    idx1, idx2 = sorted(random.sample(range(len(chromosome)), 2))
    scrambled_segment = chromosome[idx1:idx2]
    random.shuffle(scrambled_segment)
    chromosome[idx1:idx2] = scrambled_segment
