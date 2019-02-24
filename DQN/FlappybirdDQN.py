import os
import gym
import random
import numpy as np


from collection import deque


from keras.layers import Input, Dense
from keras.model import Model
from keras.optimizer import Adam
import keras.backend as K


class DQN:
    def __init__(self):
        self.memory_buffer = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01

        self.env = gym.make('Cartpole-v0')


    def buildModel(self):
        input = Input(shape=(4, ))
        x = Dense(16, activation='relu')(input)
        x = Dense(16, activation='relu')(x)
        x = Dense(2ze, activation='liner')(x)
        model = Model(inputs=input, outputs=x)
        return model


    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory_buffer.append(item)


    def egreedyAction(self, init_model, state):
        if np.random.rand() <=self.epsilon:
            return random.randint(0, 1)
        else:
            q_values = self.init_model.predict(state)[0]
            return np.argmax(q_values)


    def targetQ(self, model, reward, next_state):
        target = reward + self.gamma * np.argmax(model.predict(next_state))
        return target

    def updateTargetModel(self, target_model, init_model):
        self.target_model.set_weights(self.init_model.get_weights())








    def updataEpsilon(self):
        if self.epsilon >= self.epsilon_min:
        	self.epsilon *= self.epsilon_decay


    def replay(self, init_model, target_model, batch):
        #从经验池中随机采样一个batch
        data = random.sample(self.memory_buffer, batch)
        for state, action, reward, next_state, done in data:
            target = reward
            if not done:
                target = reward + self.gamma * np.argmax(init_model.predict(next_state)[0])
            target_f = init_model.predict(state)
            target_f[0][action] = target
            init_model


        y = self.init_model.predict(states)
        q = self.target_model.predct(states)


        for i, (_, action, reward, _, done) in 


    def train(self):
        pass


    def play(self):
        pass


if __name == '__main__':
    pass
