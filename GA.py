from deap import base, creator, tools, algorithms
import numpy as np
from ANN import ANN
import tetromino as game

NUM_INDIVIDUALS = 10;
NUM_INPUTS = 204;
NUM_HIDDEN_NODES = 4;
NUM_OUTPUTS = 4;
NUM_WEIGHTS = (NUM_INPUTS + 1) * (NUM_HIDDEN_NODES) + (NUM_HIDDEN_NODES + 1) * (NUM_OUTPUTS);

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) 
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_real",np.random.uniform,0,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_real, NUM_WEIGHTS)
toolbox.register("population", tools.initRepeat,list, toolbox.individual, NUM_INDIVIDUALS)


#Uses a game's score as the fitness
def eval(individual):
	return game.getScore(),

#creates evaluation, mutation, crossover, and selection
toolbox.register("evaluate", eval)
toolbox.register("select",tools.selTournament, tournsize = 2)
toolbox.register("mate", tools.cxBlend, alpha = 1.0)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=.3)
toolbox.register("map",map)

#Start genetic algorithm
CXPB, MUTPB, NGEN = 0.5, 0.03, 500
pop = toolbox.population()
fits = toolbox.map(toolbox.evaluate,pop)

#Create the computers to play tetris
for ind in pop:
	ann = ANN(NUM_INPUTS, NUM_HIDDEN_NODES, NUM_OUTPUTS, ind)
	#Runs the game to get a score for the individual
	my_game = game.main(ann)

#Set the fitness for the initial population
for ind,fit in zip(pop,fits):
	ind.fitness.values = fit
	
for g in range(NGEN):
	pop = toolbox.select(pop, k=len(pop))
	for ind in pop:
		ann = ANN(NUM_INPUTS, NUM_HIDDEN_NODES, NUM_OUTPUTS, ind)
		#Runs the game to get a score for the individual
		my_game = game.main(ann)
		ind.fitness.values = eval
	pop = algorithms.varAnd(pop,toolbox,CXPB,MUTPB)
	fits = toolbox.map(toolbox.evaluate, pop)
	for ind, fit in zip(pop,fits):
		ind.fitness.values = fit
best = tools.selBest(pop,k=1)
print best
	
	
