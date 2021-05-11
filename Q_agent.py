import argparse
import retro
import random
import cv2
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import euclidean_distances

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


#HYPERPARAMETERS
replacementMargin=2
stateStored=250
forwardBonus=5
stallPenalty=5
stallRadius=5
stallReset=0.99
penalizeJumpStall=False



#encode actions into action array
actions=np.zeros([7,12])
right=np.zeros([12])
left=np.zeros([12])
jump=np.zeros([12])
crouch=np.zeros([12])
right =[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
left= [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
jump= [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
crouch=[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
actions[0]=right
actions[1]=np.add(right,jump)
actions[2]=left
actions[3]=np.add(left,jump)
actions[4]=jump
actions[5]=crouch
#actions[6]=right
recentStates=np.zeros([stateStored,4])
Qarray=np.zeros([1000,7])
QarrayReps=np.zeros([1000,7])
recentCounter=0
curState=0
imgarray = []
inx, iny, inc = env.observation_space.shape
maxpos=0

inx = int(inx/8)
iny = int(iny/8)
#stores the IMG state for up to 1,000 states
stateIMG=np.zeros([1000,1120])
zeroRow=np.zeros([1120])
#Size of each cluster

clusterSize=np.zeros([1000])
states=0
#default policy of going forward

Policy=np.zeros([1000,12])
#Set Policy for each state to default
for i in range(len(Policy)):
    Policy[i]=actions[0]
    #print(i)
curState=0


try:
    while True:
        ob = env.reset()
        t = 0
        totrew = [0] * args.players
        #start with initial action for data
        ob, rew, done, info = env.step(actions[6])
        #captures environment
        xpos = info['x']
        totalRew=0

        while True:
            if(states<1000):
                act=random.randint(0, 13)
                if(act>6):
                    act=0
            else:
                #print(np.max(Qarray[curState]))
                act=np.argmax(Qarray[curState])
            if(recentCounter<stateStored):
                QarrayReps[curState][act]+=1
                recentStates[recentCounter][0]=curState
                recentStates[recentCounter][1]=act
                recentStates[recentCounter][2]=xpos
                recentStates[recentCounter][3]=totalRew
                recentCounter+=1
            else:
                stall=0
                MoveStall=0
                for i in range(stateStored):
                    if (abs(xpos-recentStates[i][2])<=stallRadius):
                        MoveStall+=1
                        if(act==recentStates[i][1]):
                            stall+=1
                            if(not penalizeJumpStall and (act==1 or act==4)):
                                #dont give stall penalty if not penalizeJumpStall and jumping
                                stall-=1

                #print(stall)
                if(MoveStall>=stateStored*stallReset and states>=100):
                    # if stalling reset
                    #print("reset")
                    zero=np.zeros([7])
                    Qarray[curState]=zero
                    QarrayReps[curState]=zero
                for i in range(stateStored):
                    #print(recentStates[i][0])
                    #print(recentStates[i][1])
                    reward=(xpos-recentStates[i][2])
                    if(recentStates[i][1]==0 or recentStates[i][1]==1):
                        reward+=forwardBonus
                    if (abs(reward)<=stallRadius):
                        reward-=stallPenalty*stall
                        if(act==6 or act==5):
                            reward-=stallPenalty*stall
                    #print(reward)
                    reps=QarrayReps[int(recentStates[i][0])][int(recentStates[i][1])]
                    if(reps<0):
                        print(reps)
                    Qarray[int(recentStates[i][0])][int(recentStates[i][1])]=(reps*Qarray[int(recentStates[i][0])][int(recentStates[i][1])]+reward)/(reps+1)
                    #print( Qarray[int(recentStates[i][0])][int(recentStates[i][1])])
                update=recentCounter%stateStored
                recentStates[update][0]=curState
                recentStates[update][1]=act
                recentStates[update][2]=xpos
                recentStates[update][3]=totalRew
                recentCounter+=1
                QarrayReps[curState][act]+=1
            #print(xpos)
            ac=actions[act]
            #ac = Policy[curState]
            #print(Policy[curState])
            #print("\n")
            
            #if(random.randint(0,2)==0):
                #adds variance for more interesting runs
            #    ac=env.action_space.sample()
            #print(ac)
            #print("\n")
            #print("\n")
            #print("\n")
            ob, rew, done, info = env.step(ac)
            xpos = info['x']
            if maxpos < xpos:
                maxpos=xpos
            t += 1
            if t % 5 == 0:
                #captures environment
                ob = cv2.resize(ob, (inx, iny))
                ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
                ob = np.reshape(ob, (inx, iny))

                #cv2.imshow('main', scaledimg)
                #cv2.waitKey(1)
                #xpos_end = info['screen_x_end']
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
                    curState=states
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
                    #Check if new scan is more different from closest scan then two existing scans by replacement margin
                    ranstate=random.randint(0,999)
                    formattedRan=np.zeros([2,1120])
                    formattedRan[0]=stateIMG[ranstate]
                    #Set Random Center to zero in order to find next closest cluster center
                    stateIMG[ranstate]=zeroRow
                    closestRan, _ = pairwise_distances_argmin_min(formattedRan, stateIMG)
                    stateIMG[ranstate]=formattedRan[0]
                    formattedClosestRan=np.zeros([2,1120])
                    formattedClosestRan[0]=stateIMG[closestRan[0]]
                    formattedClosest=np.zeros([2,1120])
                    formattedClosest[0]=stateIMG[closest[0]]
                    closestNewDist=euclidean_distances(formatted,formattedClosest)
                    closestRandDist=euclidean_distances(formattedRan, formattedClosestRan)
                    #print(closestNewDist)
                    #print("\n")
                    #print(closestRandDist)
                    #print("\n \n")
                    if(closestRandDist[0][0]*replacementMargin<closestNewDist[0][0]):
                        #if the distance between the randomly selected cluster center and the closest center to that cluster times the replacementMargin
                        #  is less then the distance between the new image and the closest center combine random and closest and make the new
                        # image a cluster center
                        #print("replace")
                        stateIMG[closestRan[0]]=(clusterSize[closestRan[0]]*stateIMG[closestRan[0]]+clusterSize[ranstate]*stateIMG[ranstate])/(clusterSize[ranstate]+clusterSize[closestRan[0]]) 
                        #update Q-array on replace
                        Qarray[closestRan[0]]=(clusterSize[closestRan[0]]*Qarray[closestRan[0]]+clusterSize[ranstate]*Qarray[ranstate])/(clusterSize[ranstate]+clusterSize[closestRan[0]])
                        QarrayReps[closestRan[0]]+=QarrayReps[ranstate]
                        clusterSize[closestRan[0]]+=clusterSize[ranstate]
                        Qarray[ranstate]=np.zeros([7])
                        
                        stateIMG[ranstate]=formatted[0]
                        clusterSize[ranstate]=1
                        curState=ranstate
                    else:
                        #Preform K-Means clustering
                        stateIMG[closest[0]]=(clusterSize[closest[0]]*stateIMG[closest[0]]+imgarray)/(1+clusterSize[closest[0]])
                        clusterSize[closest[0]]+=1
                        curState=closest[0]
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
                    totalRew=totrew[i]
            if done:
                env.render()
                try:
                    if verbosity >= 0:
                        if args.players > 1:
                            print("done! total reward: time=%i, reward=%r, Max position attained= %d" % (t, totrew,maxpos))
                        else:
                            print("done! total reward: time=%i, reward=%d, Max position attained=%d" % (t, totrew[0],maxpos))
                        input("press enter to continue")
                        #ob, rew, done, info = env.step([False, False, False, True, False, False, False, False, False, False, False, False])
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                break
except KeyboardInterrupt:
    exit(0)
