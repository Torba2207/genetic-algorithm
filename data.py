import pandas as pd

def get_small():
    knapsack = pd.read_csv('knapsack-small.csv')
    return knapsack, 6, 6

def get_big():
    knapsack = pd.read_csv('knapsack-big.csv')
    return knapsack, 500, 500


def get_extra_small():
    knapsack=pd.read_csv('knapsack-extra-small.csv')
    return knapsack, 3, 3