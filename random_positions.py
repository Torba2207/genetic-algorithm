from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np

from data import *

def generate_position(width, height, knapsack_max_width, knapsack_max_height):
    return random.choice([True, False]), random.choice(range(knapsack_max_width - width)), random.choice(range(knapsack_max_height - height)), random.choice([True, False])
def initial_population(items, individual_size, population_size):
    return [[generate_position(items["Width"][i], items["Height"][i], knapsack_max_width, knapsack_max_height) for i in range(individual_size)] for _ in range(population_size)]

def fitness(items, individual):
    x_ranges = []
    y_ranges = []
    value = 0

    for i in range(len(individual)):
        included, x, y, isRotated = individual[i]
        width, height = items["Width"][i], items["Height"][i]

        if isRotated:
            width, height = height, width

        if included:
            for x_range in x_ranges:
                for y_range in y_ranges:
                    if not ((x_range[0] >= x + width or x_range[1] <= x) or (y_range[0] >= y + height or y_range[1] <= y)):
                        return 0

            x_ranges.append((x, x + width))
            y_ranges.append((y, y + height))
            value += items["Value"][i]

    return value

def population_best(items, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_wheel_selection(items, n_selection, population):
    sum_f = sum([fitness(items, individual) for individual in population])
    selection_probs = [fitness(items, individual) / sum_f for individual in population]

    selection = [None] * n_selection
    for i in range(n_selection):
        selection[i] = population[np.random.choice(len(population), p=selection_probs)]

    return selection

def elitisme(n_elite, population, items):
    population_fitness_indices = np.argsort([fitness(items, individual) for individual in population])[::-1]
    return [population[population_fitness_indices[i]] for i in range(n_elite)]

def crossover(parent1, parent2):
    crossover_point = random.randrange(len(parent1))
    child1 = parent1[crossover_point:] + parent2[:crossover_point]
    child2 = parent2[crossover_point:] + parent1[:crossover_point]
    return child1, child2

def mutate(items, population, knapsack_max_width, knapsack_max_height, pMut = 0.01):
    for j in range(len(population)):
        individual = population[j]

        for i in range(len(individual)):
            height, width = items["Height"][i], items["Width"][i]
            if random.random() < pMut:
                k = random.randrange(len(individual))
                individual[k] = (not individual[i][0], random.choice(range(knapsack_max_width - width)), random.choice(range(knapsack_max_height - height)), not individual[i][3])

items, knapsack_max_width, knapsack_max_height = get_big()

population_size = 100
generations = 30
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []

population = initial_population(items, len(items), population_size)
sum_f = sum([fitness(items, individual) for individual in population])

at = 0
while sum_f == 0:
    at += 1
    print(at)
    population = initial_population(items, len(items), population_size)
    sum_f = sum([fitness(items, individual) for individual in population])

print(items)

for gen in range(generations):
    print(gen)
    population_history.append(population)
    selection = roulette_wheel_selection(items, n_selection, population)
    elite = elitisme(n_elite, population, items)
    next_generation = []
    next_generation_size = population_size - n_selection - n_elite

    while (len(next_generation)) < next_generation_size:
        parent1, parent2 = random.sample(selection, 2)
        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([child1, child2])

    mutate(items, next_generation, knapsack_max_width, knapsack_max_height)
    next_generation += elite + selection

    population = next_generation

    best_individual, best_individual_fitness = population_best(items, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], list(map(lambda x: x[0], best_solution)))))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

for i in range(len(best_solution)):
    included, x, y, isRotated = best_solution[i]
    if included:
        print(items["Name"][i])


# visualisation
fig = plt.gcf()
ax = fig.gca()
for i in range(len(best_solution)):
    included, x, y, isRotated = best_solution[i]
    width, height = items["Width"][i], items["Height"][i]

    if isRotated:
        width, height = height, width

    if included:
        col = (random.random(), random.random(), random.random())
        ax.add_patch(plt.Rectangle((x, y), width, height, color=col))
        print(f"{x} {y}: {items['Name'][i]} {col}. Is rotated: {isRotated}")

plt.xlim([0, knapsack_max_width])
plt.ylim([0, knapsack_max_height])
plt.show()

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
