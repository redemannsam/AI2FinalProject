import argparse
import retro
import random
import cv2
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min

parser = argparse.ArgumentParser()
parser.add_argument('--game', default='Airstriker-Genesis', help='the name or path for the game to run')
parser.add_argument('--state', help='the initial state file to load, minus the extension')
parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
parser.add_argument('--obs-type', '-o', default='image', choices=['image', 'ram'], help='the observation type, either `image` (default) or `ram`')
parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
args = parser.parse_args()

obs_type = retro.Observations.IMAGE if args.obs_type == 'image' else retro.Observations.RAM
env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players, obs_type=obs_type)
verbosity = args.verbose - args.quiet





imgarray = []
inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)
#stores the IMG state for up to 1,000 states
stateIMG=np.zeros([1000,1120])
#Size of each cluster
clusterSize=np.zeros([1000])
states=0
#default policy of going forward
default =[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
Policy=np.zeros([1000,12])
#Set Policy for each state to default
for i in range(len(Policy)):
    Policy[i]=default
    #print(i)
curState=0
try:
    while True:
        ob = env.reset()
        t = 0
        totrew = [0] * args.players
        while True:
            ac = Policy[curState]
            #print(Policy[curState])
            #print("\n")
            
            if(random.randint(0,2)==0):
                #adds variance for more interesting runs
                ac=env.action_space.sample()
            #print(ac)
            #print("\n")
            #print("\n")
            #print("\n")
            ob, rew, done, info = env.step(ac)
            t += 1
            if t % 10 == 0:
                #captures environment
                ob = cv2.resize(ob, (inx, iny))
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
                ob = np.reshape(ob, (inx, iny))

                #cv2.imshow('main', scaledimg)
                #cv2.waitKey(1)

                #convert screenshot into 1D array of values
                for x in ob:
                    for y in x:
                        imgarray.append(y)

                #print(len(imgarray))
                #print("\n")
                if(states<1000):
                    #If states array not full add to states array
                    stateIMG[states]=imgarray
                    #print(stateIMG[states])
                    #print("\n")
                    clusterSize[states]+=1
                    states+=1
                else:
                    #formatt states array
                    formatted=np.zeros([2,1120])
                    formatted[0]=imgarray
                    #find closest point in states array
                    closest, _ = pairwise_distances_argmin_min(formatted, stateIMG)
                    curState=closest[0]
                    #print(closest[0])
                    #print("\n")
                    #Preform K-Means clustering
                    stateIMG[closest[0]]=(clusterSize[closest[0]]*stateIMG[closest[0]]+imgarray)/(1+clusterSize[closest[0]])
                    clusterSize[closest[0]]+=1
                    #print(stateIMG[closest[0]])
                    #print("\n")

                imgarray.clear()        
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                env.render()
            if args.players == 1:
                rew = [rew]
            for i, r in enumerate(rew):
                totrew[i] += r
                if verbosity > 0:
                    if r > 0:
                        print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                    if r < 0:
                        print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
            if done:
                env.render()
                try:
                    if verbosity >= 0:
                        if args.players > 1:
                            print("done! total reward: time=%i, reward=%r" % (t, totrew))
                        else:
                            print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                break
except KeyboardInterrupt:
    exit(0)
