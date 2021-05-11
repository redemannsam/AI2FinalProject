import retro
import numpy as np
import neat
import cv2
import pickle
import time

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1')
img = []
totaltime = 0.0

#runs neat, evaluates genome's fitness, checks if goal has been reached
def eval_genomes(genomes, config):
    global totaltime
    bestGenome = None
    for genome_id, genome in genomes:
        if genome.fitness == None:
            genome.fitness = 0
        if bestGenome == None or bestGenome.fitness < genome.fitness:
                bestGenome = genome
        #check how long training has been running; if training time is done, save best genome
        if(totaltime >= 900):
            with open('winner.pkl', 'wb') as output:
                pickle.dump(bestGenome, output, 1)
            print(bestGenome.fitness)
            exit()
        start = time.time()
        ob = env.reset()
        ac = env.action_space.sample()

        x, y, c = env.observation_space.shape

        x = int(x/8)
        y = int(y/8)

        #create neural network for given genome
        net = neat.nn.RecurrentNetwork.create(genome, config)

        current_max_fitness = 0
        fitness_current = 0
        counter = 0
        xpos = 0
        xpos_max = 0

        done = False
        #cv2.namedWindow("main", cv2.WINDOW_NORMAL)
        while not done:
            #env.render()
            #scaledimg = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #scaledimg = cv2.resize(scaledimg, (iny, inx))

            #take screenshot of screen to use as input for neural net
            ob = cv2.resize(ob, (x, y))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (x, y))

            #cv2.imshow('main', scaledimg)
            #cv2.waitKey(1)

            #convert screenshot into 1D array of values
            for x_pos in ob:
                for y_pos in x_pos:
                    img.append(y_pos)

            #get output from neural network given screenshot input
            output = net.activate(img)
            
            #agent reacts based on the output of the neural network for that frame
            ob, rew, done, info = env.step(output)
            img.clear()

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
                print("ID: " + str(genome_id) + " fitness: " + str(fitness_current) + " total time: " + str(totaltime))

            #update's genome's fitness
            genome.fitness = fitness_current
        end = time.time()
        totaltime += end - start

#NEAT neural network configuration (default)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'config-feedforward')

#create population of genomes based on config file
population = neat.Population(config)

#print statistics for training to console
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)

#save a checkpoint for current genomes every generation
population.add_reporter(neat.Checkpointer(10))

#run the training program
winner = population.run(eval_genomes)

#save neural network that results in the agent reaching the goal
with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)
