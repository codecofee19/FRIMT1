# game.py
import pygame
import math
import time
import util
import random
import config as c
from deap import algorithms, base, creator, tools
from agent import Agent as A
from target import Target as T

class Brain:

	n_hiddenLayers = 1

	
	def __init__(self, individual):
		
		self.weights = []
		for i in range(len(individual)):
			self.weights.append(individual[i])
		self.layers = []
		for i in range(Brain.n_hiddenLayers):
			self.layers.append(neuronLayer())
		self.layers.append(neuronLayer())
		self.layers[-1].n_neurons = 2 #This is the output layer
		index = 0
		for i in range(len(self.layers)):
			currentLayer = self.layers[i]
			for j in range(currentLayer.n_neurons):
				end_index = index
				if i==0:
					end_index += 5
				else:
					end_index += self.layers[i-1].n_neurons+1
				currentLayer.neurons.append(Neuron(individual[index:end_index]))
                		index = end_index
	
	def evaluate(self, givenInputs):

		outputs = []
		inputs = []
		for i in range(len(givenInputs)):
		    inputs.append(givenInputs[i])
		inputs.append(1)
		for i in range(self.layers[0].n_neurons):
		    currentNeuron = self.layers[0].neurons[i]
		    outputs.append(currentNeuron.totalOutput(inputs))

		for i in range(1,len(self.layers)):
		    new_outputs = []
		    for j in range(self.layers[i].n_neurons):
		        currentNeuron = self.layers[i].neurons[j]
		        new_outputs.append(currentNeuron.totalOutput(outputs))
		    outputs = new_outputs

		return outputs



			
		

class Neuron:
	
	def __init__(self, inputWeights):
		self.weights = []
		for i in range(len(inputWeights)):
			self.weights.append(inputWeights[i])
        	self.output = 0

    	def totalOutput(self, inputs):
        	totalInput = 0
		for i in range(len(inputs)):
		    totalInput += self.weights[i]*inputs[i]
		totalInput += self.weights[-1]
		return sigmoid(totalInput)



class neuronLayer:

	
	def __init__(self):
		self.n_neurons = 3
		self.neurons = []
		

def sigmoid(x):
	
	return 1/(1+math.exp(-x))

class Game:
    def __init__(self):
        # pygame setup
        pygame.init()
        pygame.display.set_caption(c.game['g_name'])

        self.clock      = pygame.time.Clock()
        self.display    = pygame.display.set_mode(
                            (c.game['width'], c.game['height']))

        self.agents     = []
        self.targets    = [T() for _ in range(c.game['n_targets'])]
        self.generation = 0

        # save terminal
        print "\033[?47h"

    # add an agent with nnet argument
    def add_agent(self, nnet):
        self.agents.append(A(len(self.agents), nnet))

    def reset(self):
        self.agents = []

    # find an agent with weights argument
    def get_ind_fitness(self, ind):
        for a in self.agents:
            for i,weight in enumerate(a.brain.weights):
                if weight != ind[i]:
                    continue
                return (a.fitness,)
        return None

    def game_loop(self, display=True):
        for i in range(c.game['g_time']):

            self.game_logic()

            if i % c.game['delay'] == 0: self.update_terminal()
            if display: self.process_graphic()

        return [a.fitness for a in self.agents]

    def game_logic(self):
        for a in self.agents:

            a.update(self.targets)

            if a.check_collision(self.targets) != -1:
                self.targets[a.t_closest].reset()
                a.fitness += 1

        self.agents = util.quicksort(self.agents)
	
	# shows graphics of the game using pygame
    def process_graphic(self):
        self.display.fill((0xff, 0xff, 0xff))

        for t in self.targets:
            t_img = pygame.image.load(c.image['target']).convert_alpha()
            self.display.blit(t_img, (t.position[0], t.position[1]))

        if len(self.agents) == c.game['n_agents']:
            for i in range(c.game['n_best']):
                a_img = pygame.transform.rotate(
                    pygame.image.load(c.image['best']).convert_alpha(),
                    self.agents[i].rotation * -180 / math.pi)
                self.display.blit(a_img, (self.agents[i].position[0],
                                        self.agents[i].position[1]))

            for i in range(c.game['n_best'], c.game['n_agents']):
                a_img = pygame.transform.rotate(
                    pygame.image.load(c.image['agent']).convert_alpha(),
                    self.agents[i].rotation * -180 / math.pi)
                self.display.blit(a_img, (self.agents[i].position[0],
                                        self.agents[i].position[1]))
        else:
            for a in self.agents:
                a_img = pygame.transform.rotate(
                    pygame.image.load(c.image['best']).convert_alpha(),
                                    a.rotation * -180 / math.pi)
                self.display.blit(a_img, (a.position[0], a.position[1]))

        pygame.display.update()
        self.clock.tick(c.game['fps'])

    def update_terminal(self):
        print "\033[2J\033[H",
        print c.game['g_name'],
        print "\tGEN.: " + str(self.generation),
        print "\tTIME: " + str(time.clock()) + '\n'
	
	agents_string = ""

        '''for a in self.agents:
            print "AGENT " + repr(a.number).rjust(2) + ": ",
            print "FITN.:" + repr(a.fitness).rjust(5)
	'''
	index = 0
	for a in self.agents:
		agents_string += "AGENT " + repr(a.number).rjust(2) + ": " + "FITN.:" + repr(a.fitness).rjust(5)
		if index%2==1:
			agents_string+="\n"
		else:
			agents_string += "      "
		index+=1
	print agents_string

# run the game without GA and ANN
def fitnessFunction(game,individual):
    
    return game.get_ind_fitness(individual)

def mutate(individual, indpb = .05):
        p = indpb
        for i in range(len(individual)):
            if p>=random.random():
                n = random.uniform(-1,1)
                individual[i] += n
        return individual,

def crossOver(individual1, individual2, alpha=.5):
    p = .1
    gamma = (1.+2.*alpha)*random.random()-alpha
    for i in range(len(individual1)):
	first = individual1[i]
	second = individual2[i]
	if p>random.uniform(0,1):
		individual1[i] = gamma*second + (1.-gamma)*first
        	individual2[i] = gamma*first + (1.-gamma)*second
    return individual1,individual2
	

def createPop(initRepeat, nType, individual, game):

	population = initRepeat(nType,individual,n=c.game['n_agents'])
	if len(game.agents)==c.game['n_agents']:
        	game.reset()
	for i in range(len(population)):
		game.add_agent(Brain(population[i]))
	
	game.game_loop(display=False)
	
	game.generation +=1

	for a in game.agents:
            print "AGENT " + repr(a.number).rjust(2) + ": ",
            print "FITN.:" + repr(a.fitness).rjust(5)
 
	return population

def modifiedEASimple(population, toolbox, cxpb, mutpb, ngen, game, stats=None,
             halloffame=None, verbose=__debug__):

    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.
    
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population and a :class:`~deap.tools.Logbook`
              with the statistics of the evolution.
    
    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution (if
    any). The logbook will contain the generation number, the number of
    evalutions for each generation and the statistics if a
    :class:`~deap.tools.Statistics` if any. The *cxpb* and *mutpb* arguments
    are passed to the :func:`varAnd` function. The pseudocode goes as follow
    ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.
    
    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.
    
    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # Replace the current population by the offspring
        population[:] = offspring
	
	# MODIFIED BLOCK: Adds new agents based on the new offspring and goes through a game loop to set 		fitness values
	game.reset()
	for i in range(len(population)):
		game.add_agent(Brain(population[i]))
	game.game_loop(display=False)
	game.generation += 1
	
	
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream        

    return population, logbook

if __name__ == '__main__':
    g = Game()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
	
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.uniform,-5,5)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=23)
    toolbox.register("population", createPop, tools.initRepeat, list, toolbox.individual, g)
    toolbox.register("evaluate", fitnessFunction, g)
    toolbox.register("mate", crossOver)
    toolbox.register("mutate", mutate, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population()
    result = modifiedEASimple(pop, toolbox, cxpb=.5, mutpb=.2, ngen=50, game = g, verbose = False)
	
    g.reset()

    for i in range(c.game['n_agents']):
        g.add_agent(Brain(result[0][i]))
    
    g.game_loop()
    pygame.quit()
