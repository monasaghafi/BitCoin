import pandas as pd
import random
import pygad


def get_data():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    return train, test


def fitness_func(ga_instance, solution, solution_idx):
    train, test = get_data()
    fitness = 0
    for i in range(len(train)):
        function_inputs = train.iloc[i, 1:11]  # indexLocation
        s = 0  # final result=b0+b1x1+b2x2+...+b10x10
        for j in range(1, 11):
            s += solution[j] * function_inputs[j - 1]
        output = s + solution[0]
        main_out = train.iloc[i, 11]
        err = (main_out - output) ** 2
        fitness += err
    return 1 / fitness


def test(solution):
    train, test = get_data()
    fitness = 0
    for i in range(len(test)):
        function_inputs = test.iloc[i, 1:11]  # indexLocation
        s = 0  # final result=b0+b1x1+b2x2+...+b10x10
        for j in range(1, 11):
            s += solution[j] * function_inputs[j - 1]
        output = s + solution[0]
        main_out = test.iloc[i, 11]
        err = (main_out - output) ** 2
        fitness += err
    return fitness


if __name__ == "__main__":
    train_data, test_data = get_data()

    fitness_function = fitness_func

    num_generations = 50
    num_parents_mating = 4

    sol_per_pop = 10
    num_genes = 11

    init_range_low = -2
    init_range_high = 5

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 10

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_function,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes)
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
    print(test(solution))

