#!/usr/bin/env python3
import numpy as np
import gym
import gym_minigrid
import matplotlib 
import random
matplotlib.use('TkAgg')
env = gym.make('MiniGrid-Empty-8x8-v0')


total_episodes = 200
max_steps = 200
alpha = 0.3
gamma = 0.9

Q = np.zeros((6, 6, 4, 3))
x, y = env.agent_pos[0], env.agent_pos[1]
dir = env.agent_dir

def choose_action(x, y, dir, epsilon):
    choice= [np.random.randint(0, 3), np.argmax(Q[x][y][dir])]
    p = [epsilon, 1-epsilon]
    return random.choices(choice, p)[0]


def update(x1, y1, dir1, x, y, dir, action, action2, reward):
    predict = Q[x][y][dir][action]
    target = reward + gamma * Q[x1][y1][dir1][action2]
    Q[x][y][dir][action] = Q[x][y][dir][action] + alpha*(target - predict)

reward = 0
for episode in range(total_episodes):
    epsilon = 100/(100 + episode)
    print(episode)
    t = 0
    env.reset()
    action1 = choose_action(x, y, dir, epsilon)
    x1, y1 = env.agent_pos[0], env.agent_pos[1]
    dir1 = env.agent_dir
    
    while t < max_steps:
        env.render()
        obs, reward, done, info = env.step(action1)
        action2 = choose_action(x1, y1, dir1, epsilon)
        update(x1, y1, dir1, x, y, dir, action1, action2, reward)
        x, y = x1, y1
        dir = dir1
        action1 = action2
        t += 1
        reward += 1
        if done:
            break
        
    






                    




        







