import retro
import numpy as np
import neat
import cv2
import pickle
import time

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
imgarray = []

#runs neat, evaluates genome's fitness, checks if goal has been reached
def run_genome(genomes, config):
    ob = env.reset()
    ac = env.action_space.sample()

    inx, iny, inc = env.observation_space.shape

    inx = int(inx/8)
    iny = int(iny/8)

    #create neural network for given genome
    net = neat.nn.RecurrentNetwork.create(genome, config)

    current_max_fitness = 0
    fitness_current = 0
    counter = 0
    xpos = 0
    xpos_max = 0

    done = False
    while not done:
        env.render()

        #take screenshot of screen to use as input for neural net
        ob = cv2.resize(ob, (inx, iny))
        ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
        ob = np.reshape(ob, (inx, iny))

        #convert screenshot into 1D array of values
        for x in ob:
            for y in x:
                imgarray.append(y)

        #get output from neural network given screenshot input
        nnOutput = net.activate(imgarray)
            
        #agent reacts based on the output of the neural network for that frame
        ob, rew, done, info = env.step(nnOutput)
        imgarray.clear()

        xpos = info['x']
        xpos_end = info['screen_x_end']

        #increase reward as agent continues moving rightward
        if xpos > xpos_max:
            fitness_current += 1
            xpos_max = xpos

        #if agent reached the end of the level, goal has been reached
        if xpos == xpos_end and xpos > 100:
            fitness_current += 10000
            done = True
                
        #if the agent does not continue moving right, add 1 to counter (otherwise reset to 0)
        if fitness_current > current_max_fitness:
            current_max_fitness = fitness_current
            counter = 0
        else:
            counter += 1

        #if the agent does not make rigthward progress within 250 frames, move on to next genome
        if done or counter == 250:
            done = True
            print("fitness: " + str(fitness_current))

        #update's genome's fitness
        genome.fitness = fitness_current

#NEAT neural network configuration (default)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')
genome = None
with open('winner.pkl', 'rb') as agent:
    genome = pickle.load(agent)
run_genome(genome, config)
