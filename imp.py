#!/usr/bin/env python3
import numpy as np
import gym
import gym_minigrid
import matplotlib 
import random
matplotlib.use('TkAgg')
env = gym.make('MiniGrid-Empty-8x8-v0')

def choose_action(x, y, dir, epsilon):
    choice= [np.random.randint(0, 3), np.argmax(Q[x][y][dir])]
    p = [epsilon, 1-epsilon]
    return random.choices(choice, p)[0]


total_episodes = 200
max_steps = 200
alpha = 0.3
gamma = 0.9
env.reset()

Q = np.zeros((6, 6, 4, 3))

for episode in range(200):
    print(episode)
    E = np.zeros((6, 6, 4, 3))
    x, y = env.agent_pos[0], env.agent_pos[1]
    dir = env.agent_dir

    done = False

    while not done:
        epsilon = 100/(100 + epsilon)
        action = choose_action(x, y, dir, epsilon)
        obs, reward, done, info = env.step(action)
        x1, y1 = env.agent_pos[0], env.agent_pos[1]
        dir1 = env.agent_dir

        target = reward + (gamma*Q[x1][y1][dir1][np.argmax(Q[x1][y1][dir1])]) - Q[x][y][dir][action]

        E[x][y][dir][action] += 1

        for a in range(len(Q)):
            for b in range(len(Q[a])):
                for c in range(len(Q[a][b])):
                    max_a = 0 
                    for ac in range(3):
                        Q[a][b][c][ac] = Q[a][b][c][ac] + (alpha*target*E[a][b][c][ac])
                        E[a][b][c][ac] = gamma*0.8*E[a][b][c][ac]
                        if Q[a][b][c][ac] > Q[a][b][c][max_a]:
                            max_a = ac