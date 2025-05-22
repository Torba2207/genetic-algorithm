from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy
import numpy as np
from data import *

def generate_position(knapsack_max_width):
    return random.choice([True, False]), random.choice(range(knapsack_max_width)), random.choice([True, False])

def random_indeces(n):
    res = range(n)
    res = list(res)
    random.shuffle(res)
    return res
def initial_population(items, individual_size, population_size):
    return [([generate_position(knapsack_max_width) for i in range(individual_size)], random_indeces(len(items))) for _ in range(population_size)]

def fitness(items, individual, knapsack_max_height, knapsack_max_width):
    heights = [0] * knapsack_max_width
    value = 0

    items_params, order = individual

    for i in range(len(items)):
        num = order[i]
        included, xPos, isRotated = items_params[num]
        width, height = items["Width"][num], items["Height"][num]

        if isRotated:
            width, height = height, width

        if included:
            max_height = 0

            if xPos + width > knapsack_max_width:
                return 0

            for x in range(xPos, xPos + width):
                max_height = max(max_height, heights[x])

            if max_height + height > knapsack_max_height:
                return 0

            for x in range(xPos, xPos + width):
                heights[x] = max_height + height

            value += items["Value"][num]

    return value

def population_best(items, population, knapsack_max_height):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, individual, knapsack_max_height, knapsack_max_width)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_wheel_selection(items, n_selection, population, knapsack_max_height):
    sum_f = sum([fitness(items, individual, knapsack_max_height, knapsack_max_width) for individual in population])
    selection_probs = [fitness(items, individual, knapsack_max_height, knapsack_max_width) / sum_f for individual in population]

    selection = [None] * n_selection
    for i in range(n_selection):
        selection[i] = population[np.random.choice(len(population), p=selection_probs)]

    return selection

def elitisme(n_elite, population, items, knapsack_max_height):
    population_fitness_indices = np.argsort([fitness(items, individual, knapsack_max_height, knapsack_max_width) for individual in population])[::-1]
    return [population[population_fitness_indices[i]] for i in range(n_elite)]


def crossover_arrays(array1, array2, crossover_point):
    if crossover_point == 0:
        return array1, array2

    child1_1 = array1[crossover_point:]
    child2_1 = array2[crossover_point:]

    child1_2 = array1[:crossover_point]
    child2_2 = array2[:crossover_point]

    child1 = child1_1 + [child1_2[i] for i in numpy.argsort(child2_2)]
    child2 = child2_1 + [child2_2[i] for i in numpy.argsort(child1_2)]

    return child1, child2
def crossover(parent1, parent2):
    crossover_point = random.randrange(len(parent1[1]))

    new_array1, new_array2 = crossover_arrays(parent1[1], parent2[1], crossover_point)

    crossover_point = random.randrange(len(parent1[0]))

    child1 = (parent1[0][crossover_point:] + parent2[0][:crossover_point], new_array1)
    child2 = (parent2[0][crossover_point:] + parent1[0][:crossover_point], new_array2)

    return child1, child2

def mutate(items, population, knapsack_max_width, knapsack_max_height, pMut = 0.01):
    for j in range(len(population)):
        params, order = population[j]

        for i in range(len(params)):
            num = order[i]
            height, width = items["Height"][num], items["Width"][num]

            if random.random() < pMut:
                included, pos_x, rotated = params[num]

                if rotated:
                    height, width = width, height

                type = random.randint(0, 4)
                if type == 0:
                    params[num] = (not included, pos_x, rotated)
                elif type == 1:
                    params[num] = (included, pos_x, not rotated)
                elif type == 2:
                    params[num] = (included, random.randint(0, knapsack_max_width - width), rotated)
                else:
                    point = random.randint(0, len(order) - 1)
                    order[i], order[point] = order[point], order[i]

                population[j] = (params, order)



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
sum_f = sum([fitness(items, individual, knapsack_max_height, knapsack_max_width) for individual in population])

print("Program started")

while sum_f == 0.0:
    population = initial_population(items, len(items), population_size)
    sum_f = sum([fitness(items, individual, knapsack_max_height, knapsack_max_width) for individual in population])

print(items)

for gen in range(generations):
    print(gen)
    population_history.append(population)
    selection = roulette_wheel_selection(items, n_selection, population, knapsack_max_height)
    elite = elitisme(n_elite, population, items, knapsack_max_height)
    next_generation = []
    next_generation_size = population_size - n_selection - n_elite

    while (len(next_generation)) < next_generation_size:
        parent1, parent2 = random.sample(selection, 2)
        child1, child2 = crossover(parent1, parent2)
        next_generation.extend([child1, child2])

    mutate(items, next_generation, knapsack_max_width, knapsack_max_height)
    next_generation += elite + selection

    population = next_generation

    best_individual, best_individual_fitness = population_best(items, population, knapsack_max_height)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], list(map(lambda x: x[0], best_solution[0])))))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

best_params, best_order = best_solution

for i in range(len(best_params)):
    included, x, isRotated = best_params[i]
    if included:
        print(items["Name"][i])


# visualisation
fig = plt.gcf()
ax = fig.gca()

heights = [0] * knapsack_max_width
print(best_order)

for i in range(len(best_params)):
    num = best_order[i]
    included, x, isRotated = best_params[num]
    width, height = items["Width"][num], items["Height"][num]

    if isRotated:
        width, height = height, width

    if included:
        max_height = 0

        for cur_x in range(x, x + width):
            max_height = max(max_height, heights[cur_x])

        y = max_height

        for cur_x in range(x, x + width):
            heights[cur_x] = max_height + height

        print(f"Size", height, width)
        col = (random.random(), random.random(), random.random())
        ax.add_patch(plt.Rectangle((x, y), width, height, color=col))
        print(f"{x} {y}: {items['Name'][num]} {col}. Is rotated: {isRotated}, num {num}, i: {i}")

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
    population_fitnesses = [fitness(items, individual, knapsack_max_height, knapsack_max_width) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()