import itertools
from itertools import compress
import random
import time
import matplotlib.pyplot as plt

from data import *


def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]


def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))


def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness


# knapsack_max_capacity - wielkość plecaka
items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20  # ilość rodziców do wybrania
n_elite = 1  # ile wziąść maxymalnych elem.

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # TODO: implement genetic algorithm
    # random.shafel (odsortowanie listy - losowe rozrzucenie el.) / choice (bez wag) / choices(tab, pram-wagi) /
    # itemtools.combinations (iloczyn kart. elementów --2-- stopnia; zwraca kombinacje - trzeba podać wielkość toopli) /
    # random.sample (pewna ilość el ze zbioru)
    # individual = osobnik
    # 6 linijek, 900 znaków...

    # 2. Wybór rodziców (selekcja ruletkowa)
    sum_f = sum([fitness(items, knapsack_max_capacity, individual) for individual in population])
    p = [(fitness(items, knapsack_max_capacity, individual)/sum_f) for individual in population]
    parents = random.choices(population, p, k=n_selection)

    # 3. Tworzenie kolejnego pokolenia
    elite = population_best(items, knapsack_max_capacity, population)[0]
    tuples = itertools.combinations(parents, 2)
    parents_tuples = random.sample([*tuples], population_size-n_elite)
    children = [[random.choice(p) for p in zip(p1, p2)] for p1, p2 in parents_tuples]

    # 4. Mutacja
    # wybór dzieci, które będziemy mutować
    rnd_mutations = random.choices(children, k=random.randint(0, n_selection))
    for child in rnd_mutations:
        # wybór ilości cech mutowanych - max to 2 cechy
        rnd_nmb = random.randint(0, 2)
        for i in range(rnd_nmb):
            idx = random.randint(0, child.__len__()-1)
            child[idx] = not child[idx]

    # Tomek
    # for i in range(n_selection):
    #     child_nmb = random.randint(0, children.__len__())
    #     idx = random.randint(0, 26)
    #     children[child_nmb][idx] = not children[child_nmb][idx]

    # 5. Aktualizacja populacji rozwiązań
    population = children + [elite]

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
