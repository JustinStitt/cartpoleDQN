import gym
from DQN import Agent
import numpy as np
import torch as T
import sys, os


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    brain = Agent(gamma = 0.99, epsilon = 1.0, batch_size = 64, n_actions = env.action_space.n,
                    input_dims = env.observation_space.shape, lr = 1e-4)


    to_load = True
    if os.path.isfile('trained_model.pt') and to_load:
        print('here')
        brain.load_checkpoint()
        brain.epsilon = brain.eps_min

    scores = []
    n_games = 3000
    score = 0

    for i in range(n_games):
        if i % 5 == 0 and i > 0:
            avg_score = np.mean(scores[-100:])
            print('episode: ', i, 'score ', score,
                        'average score %.3f' % avg_score,
                        'epsilon %.3f' % brain.epsilon)
        if i % 500 == 0 and i > 0:
            brain.save_checkpoint()#save checkpoint every 500 iterations
            print('-==SAVING CHECKPOINT==- fn: \'trained_model.pt\'')
        score = 0
        observation = env.reset()
        done = False
        while not done:
            env.render()#to draw

            action = brain.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            brain.store_transition(observation, action, reward, observation_, done)
            brain.learn()
            observation = observation_
        scores.append(score)
    brain.save_checkpoint()#save checkpoint after all episodes are done
