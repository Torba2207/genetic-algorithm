import random
from itertools import product, compress
import time
import matplotlib.pyplot as plt

from data import *

def fitness(items, individual, knapsack_width, knapsack_height):
    x_ranges = []
    y_ranges = []
    value = 0

    for i in range(len(individual)):
        included, x, y, isRotated = individual[i]
        width, height = items["Width"][i], items["Height"][i]

        if isRotated:
            width, height = height, width

        if included:
            if width + x > knapsack_width or height + y > knapsack_height:
                return 0

            for xr, yr in zip(x_ranges, y_ranges):
                if not ((xr[0] >= x + width or xr[1] <= x) or (yr[0] >= y + height or yr[1] <= y)):
                    return 0

            x_ranges.append((x, x + width))
            y_ranges.append((y, y + height))
            value += items["Value"][i]

    return value


items, max_width, max_height = get_extra_small()
print(items)

start_time = time.time()
best_solution = None
best_value = 0

for combination in product([False, True], repeat=len(items)):
    for x_pos in product(range(max_width), repeat=len(items)):
        for y_pos in product(range(max_height), repeat=len(items)):  # Corrected to max_height
            for rotation_combination in product([False, True], repeat=len(items)):
                solution = []
                for i, (include, x, y, is_rotated) in enumerate(zip(combination, x_pos, y_pos, rotation_combination)):
                    solution.append((include, x, y, is_rotated))

                solution_value = fitness(items, solution, max_width, max_height)

                if solution_value > best_value:
                    best_solution = solution
                    print(best_solution)
                    best_value = solution_value

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], list(map(lambda x: x[0], best_solution)))))
print('Best solution value:', best_value)
print('Time: ', total_time)

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

plt.xlim([0, max_width])
plt.ylim([0, max_height])
plt.show()
