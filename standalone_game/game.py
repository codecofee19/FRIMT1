# game.py
import pygame
import math
import time
import util
import random
import matplotlib.pyplot as plt
import config as c
from deap import algorithms, base, creator, tools
from agent import Agent as A
from target import Target as T

class Brain:

	n_hiddenLayers = 1

	
	def __init__(self, individual, numNeurons):
		
		self.weights = []
		for i in range(len(individual)):
			self.weights.append(individual[i])
		self.layers = []
		for i in range(Brain.n_hiddenLayers):
			self.layers.append(neuronLayer(numNeurons))

		self.layers.append(neuronLayer(2))

		index = 0
		for i in range(len(self.layers)):
			currentLayer = self.layers[i]
			for j in range(currentLayer.n_neurons):
				end_index = index
				if i==0:
					end_index += 5
				else:
					end_index += self.layers[i-1].n_neurons+1
				currentLayer.neurons.append(Neuron(self.weights[index:end_index]))
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
		
		#bias
		outputs.append(1)

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
		self.totalInput = 0
		for i in range(len(inputWeights)):
			self.weights.append(inputWeights[i])

    	def totalOutput(self, inputs):
        	totalInput = 0
		for i in range(len(inputs)):
		    totalInput += self.weights[i]*inputs[i]
		return sigmoid(totalInput)



class neuronLayer:

	
	def __init__(self, numNeurons):
		self.n_neurons = numNeurons
		self.neurons = []
		

def sigmoid(x):
	
	return 2/(1+math.exp(-x))

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
	self.num	= 0

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
        print c.game['g_name'] + str(self.num),
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
                n = random.uniform(-.5,.5)
                individual[i] += n
        return individual,

def crossOver(individual1, individual2):
    p=.1
    for i in range(len(individual1)):
	first = individual1[i]
	second = individual2[i]
	if p>random.uniform(0,1):
		individual1[i] = second
        	individual2[i] = first

    return individual1,individual2
	

def createPop(initRepeat, nType, individual, game, n_neurons, trialAverage, trialBest):

	population = initRepeat(nType,individual,n=c.game['n_agents'])
	if len(game.agents)==c.game['n_agents']:
        	game.reset()
	for i in range(len(population)):
		game.add_agent(Brain(population[i], n_neurons))
	
	game.game_loop(display=False)

	totalFitness = 0
	largest = 0
	for i in range(c.game['n_agents']):
		totalFitness += g.agents[i].fitness
		if g.agents[i].fitness>largest:
			largest = g.agents[i].fitness
	averageFitness = totalFitness/c.game['n_agents']
	trialAverage.append(averageFitness)
	trialBest.append(largest)	
	
	game.generation +=1

	for a in game.agents:
            print "AGENT " + repr(a.number).rjust(2) + ": ",
            print "FITN.:" + repr(a.fitness).rjust(5)
 
	return population

def modifiedGA(population, toolbox,  hallOfFame, n_neurons, trialAverage, trialBest, cxpb, mutpb, ngen, game):

    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population)-1)
	offspring.append(hallOfFame[0])
	
        
        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
            
        # Replace the current population by the offspring
        population[:] = offspring
	
	# MODIFIED BLOCK: Adds new agents based on the new offspring and goes through a game loop to set 		fitness values
	game.reset()
	for i in range(len(population)):
		game.add_agent(Brain(population[i], n_neurons))
	game.game_loop(display=False)

	totalFitness = 0
	largest = 0
	for i in range(c.game['n_agents']):
		totalFitness += g.agents[i].fitness
		if g.agents[i].fitness>largest:
			largest = g.agents[i].fitness
	averageFitness = totalFitness/c.game['n_agents']
	trialAverage.append(averageFitness)
	trialBest.append(largest)
	

	game.generation += 1

	     
    return population

if __name__ == '__main__':
	
	generation = []
	for i in range(101):
		generation.append(i)
	averageFitness = []
	bestFitness = []
	for i in range(3):
		trialAverage = []
		trialBest = []
		n_neurons = 5

		g = Game()
		g.num = i

		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMax)

		toolbox = base.Toolbox()
		toolbox.register("attr_bool", random.uniform,-5,5)
		toolbox.register("individual", tools.initRepeat, creator.Individual,
			     toolbox.attr_bool, n=7*n_neurons+2)
		toolbox.register("population", createPop, tools.initRepeat, list, toolbox.individual, g, n_neurons,trialAverage,trialBest)
		toolbox.register("evaluate", fitnessFunction, g)
		toolbox.register("mate", crossOver)
		toolbox.register("mutate", mutate, indpb=0.1)
		toolbox.register("select", tools.selTournament, tournsize=3)

		pop = toolbox.population()
		hallOfFame = tools.HallOfFame(1)
		hallOfFame.update(pop)
		result = modifiedGA(pop, toolbox, hallOfFame, n_neurons, trialAverage, trialBest, cxpb=.5, mutpb=.2, ngen=100, game = g)
		
		averageFitness.append(trialAverage)
		bestFitness.append(trialBest)

	plt.figure(1)
	y1 = averageFitness[0]
	y2 = averageFitness[1]
	y3 = averageFitness[2]
	plt.plot(generation,y1,"b--",label="Trial1")
	plt.plot(generation,y2,"g--",label="Trial2")
	plt.plot(generation,y3,"r--",label="Trial3")
	plt.legend(loc="upper left")
	plt.axis([0,100,0,100])
	plt.xlabel("Generation")
	plt.ylabel("Average Fitness")
	plt.title("Average Fitness vs. Generation")

	plt.figure(2)
	y4 = bestFitness[0]
	y5 = bestFitness[1]
	y6 = bestFitness[2]
	plt.plot(generation,y4,"b--",label="Trial1")
	plt.plot(generation,y5,"g--",label="Trial2")
	plt.plot(generation,y6,"r--",label="Trial3")
	plt.legend(loc="upper left")
	plt.axis([0,100,0,100])
	plt.xlabel("Generation")
	plt.ylabel("Best Fitness")
	plt.title("Best Fitnesses vs. Generation")

	plt.show()

	print("Average Fitness Data:")
	print(averageFitness)
	print("Best Fitness Data:")
	print(bestFitness)

    
    	pygame.quit()
    
	
