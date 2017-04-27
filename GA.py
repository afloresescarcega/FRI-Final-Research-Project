from deap import base, creator, tools, algorithms
import numpy as np
from ANN import ANN
import tetromino as game

NUM_INDIVIDUALS = 30;
NUM_INPUTS = 204;
NUM_HIDDEN_NODES = 4;
NUM_OUTPUTS = 6;
NUM_LAYERS = 3;
NUM_WEIGHTS = (NUM_INPUTS + 1) * (NUM_HIDDEN_NODES) + ( NUM_LAYERS - 1) * (NUM_HIDDEN_NODES+ 1) * (NUM_HIDDEN_NODES) + (NUM_HIDDEN_NODES + 1) * (NUM_OUTPUTS);

creator.create("FitnessMax", base.Fitness, weights=(1.0,1.0)) 
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_real",np.random.uniform,0,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, NUM_WEIGHTS)
toolbox.register("population", tools.initRepeat,list, toolbox.individual, NUM_INDIVIDUALS)


#Uses a game's score as the fitness
def eval(individual):
	ann = ANN(NUM_INPUTS, NUM_HIDDEN_NODES, NUM_LAYERS, NUM_OUTPUTS, individual)
	return game.main(ann)

#creates evaluation, mutation, crossover, and selection
toolbox.register("evaluate", eval)
toolbox.register("select",tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=.3)
toolbox.register("map",map)

#Start genetic algorithm
CXPB, MUTPB, NGEN = 0.05, 1, 30
pop = toolbox.population()

#Create the computers to play tetris

	
for g in range(NGEN):
	print g;
	pop = toolbox.select(pop, k=len(pop))
	for ind in pop:
		ann = ANN(NUM_INPUTS, NUM_HIDDEN_NODES, NUM_LAYERS, NUM_OUTPUTS, ind)
		#Runs the game to get a score for the individual
		ind.fitness.values = game.main(ann)
	pop = algorithms.varAnd(pop,toolbox,CXPB,MUTPB)
	pop = toolbox.select(pop, k = NUM_INDIVIDUALS)
best = tools.selBest(pop,k=1)
print best