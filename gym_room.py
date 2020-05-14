import time
import gym
from gym import spaces
import numpy as np 
from AC_room import Room
import matplotlib.pyplot as plt
from AC_room import p_controller, I_controller
import argparse
import plotly.graph_objects as go

from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, ACKTR, PPO2, SAC
from stable_baselines.common.cmd_util import make_vec_env
from stable_baselines.common.policies import FeedForwardPolicy, register_policy

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="predict", help="Define the mode train or predict")

#With PPO2, with the default parameters, the learning was fluctuating at the steady state by more than the offset value degC.
#In RL, when there is fluctuation at the steady state, I needed to decrease the learning rate (in my case from 0.001 to 0.0001) and increase the batch size by a factor of 10 
#the batch size is represented by the parameter n_steps in PPO2 from Stable Baseline, and its defalut value is 128.

#Also for the reward function, I specified to give a reward of 1 for any error less than 0.5 deg C(offset value) and
#and reduce the reward with a decreasing exponential function
class GymACRoom(gym.Env):
    metadata = {'render.modes' : ['human']}

    def __init__(self, mC=300, K=20, Q_AC_Max = 1000, simulation_time = 12*60*60, control_step = 300):

        super(GymACRoom, self).__init__()

        self.AC_sim = Room(mC=300, K=20, Q_AC_Max = 1000, simulation_time = 12*60*60, control_step = 300)
        self.time_step = control_step
        self.Q_AC_Max = Q_AC_Max
        self.action_space = spaces.Box(low = -1, high = 1, shape=(1,))
        n_obs = 1  # number of observation (dimension)
        self.observation_space = spaces.Box(low = -100, high = 100, shape =(n_obs,))
        self.observation = np.empty(n_obs)
    
    def reset(self):
        self.AC_sim.reset(T_in = np.random.randint(20, 30))
        self.iter = 0
        self.observation[0] = self.AC_sim.T_in - self.AC_sim.T_set
        return self.observation
        # self.obs [0] =dfddfdf if you have more than one obs

    def step(self, action):
        self.AC_sim.update_Tin(action=action)
        self.observation[0] = self.AC_sim.T_in - self.AC_sim.T_set
        self.iter += 1

        if self.iter >= self.AC_sim.max_iteration:
            done = True
        else:
            done = False
        
        #Reward function

        #reward NoOffSet
        reward = np.exp(-(abs(10*self.observation)))

        #reward NoOffSet
        #reward = np.exp(-(abs(self.observation)))

        #Reward with Offset
        '''
        if abs(self.observation) < 0.5:
            reward = 1
        else:
            reward = np.exp(-(abs(self.observation)-0.5))
        '''
        info = {}

        return self.observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close (self):
        pass

# Custom MLP policy of three layers of size 128 each
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[8, 8],
                                                          vf=[8, 8])],
                                           feature_extraction="mlp")



if __name__ == "__main__":

    #argparse to define the mode train or predict 
    args=parser.parse_args()
    mode = args.mode
    ####

    if mode =='train':

        env = GymACRoom()
        
        learning_rate = [0.0001]
        # Register the policy, it will check that the name is not already taken
        register_policy('CustomPolicy', CustomPolicy)
        
        for lr in learning_rate:
            model = PPO2(policy = 'CustomPolicy', env=env, verbose=1, learning_rate=lr, n_steps=1280, tensorboard_log="./AC_tensorboard/")
            model.learn(total_timesteps = 1000000)
            #model.save("AC_PPO2_LR_exp"+str(lr)+".zip")
            model.save("AC_PPO2_exp_neg_noOffset.zip")
        
        '''
        # scheduling learning rate
        #lr= 0.1
        
        model = PPO2('MlpPolicy', env, verbose=1, learning_rate=0.1, tensorboard_log="./AC_tensorboard/")
        model.learn(total_timesteps = 100000)
        model.learning_rate = 0.01
        model.learn(total_timesteps = 100000)
        model.learn(total_timesteps = 100000)
        model.learning_rate = 0.001
        model.learn(total_timesteps = 100000)
        model.save("AC_PPO2_LR_.zip")
    '''
    
    else:

        env = GymACRoom() #for RL part
        env2 = GymACRoom() #for PI controller
        

        model= PPO2.load("AC_PPO2_LR_exp.zip")
        
        obs = env.reset() #for RL part
        obs2 = env2.reset() #for PI controller
        env2.AC_sim.T_in = env.AC_sim.T_in #both RL and PI controllers start with the same initial T_in


        #plot variables definition
        n_iter = 1000
        TinRL = np.empty(n_iter)
        Tset = (env.AC_sim.T_set) * np.ones_like(TinRL)
        T_Off_high = Tset + 0.5
        T_Off_low = Tset - 0.5
        t = np.empty(n_iter)
        Tin = np.empty(n_iter)
        errorI = 0
        #
        for i in range(n_iter):
            action, _states = model.predict(obs)
            print('action is', action)

            #Storing values for plot
            t[i] = env.time_step/60 * i
            TinRL[i] = env.AC_sim.T_in
            ####
            obs, rewards, dones, info = env.step(action)
            print('states are', obs)

            error = obs2[0]
            errorI += error
            control_signal = p_controller(error) + I_controller(errorI)
            
            if abs(error) > 0.5:
                current_action = control_signal   # choose a power proportional to the gain
            else:
                current_action = 0
                
            Tin[i] = env2.AC_sim.T_in
            action2=current_action
            print('current action2', action2)
            obs2, rewards2, dones2, info2 = env2.step(action=action2)
            if dones==True:
                obs2 = env2.reset()
                obs= env.reset()
                env2.AC_sim.T_in = env.AC_sim.T_in
        plt.plot(t, TinRL, 'r--', label='RL_PPO2')
        plt.plot(t, Tin, 'b--', label='PI')
        plt.plot(t, Tset, 'g--', label='Tset')
        plt.plot(t, T_Off_high, 'k--', label='T_set_hi')
        plt.plot(t, T_Off_low, 'k--', label='T_set_lo')
        plt.xlabel('Iteration time (min)')
        plt.ylabel('Temperature (deg. C)')
        plt.legend()
        plt.show()
        
        '''
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=t, y=TinRL, name='T_in_RL', mode='markers', marker_color='rgba(70, 0, 0, .8)'))
        fig.add_trace(go.Scatter(x=t, y=Tin, name='T_in_PI', mode='markers', marker_color='rgba(200, 0, 0, .8)'))
        fig.add_trace(go.Scatter(x=t, y=Tin, name='Tset', mode='markers', marker_color='rgba(0, 0, 200, 1)'))
        
        fig.show()
        '''
        # to see the reward episod progress, get the http address by pasting the following command in the conda directory where this code is running.
        # --> tensorboard --logdir ./AC_tensorboard/








        







