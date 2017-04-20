from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import matplotlib.pyplot as plt 
import tetromino.py as game

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
creator.create("Individual", list, fitness=creator.FitnessMax)
NUM_INDIVIDS = 10 #starting out with 10

toolbox = base.Toolbox()
toolbox.register("attr_real",np.random.uniform,0,1);
toolbox.register("individual", tools.initRepeat, creator.Indivual, toolbox.attr_real, 1);
toolbox.register("population", tools.initRepeat,list, toolbox.individual, 10);

def evalFunc(individual):
	game.runGame(individual);



