import numpy as np
import random
import matplotlib.pyplot as plt
import time
#////////////////////////////////////////////////
#class example
#>>> class Complex:
#...     def __init__(self, realpart, imagpart):
#...         self.r = realpart
#...         self.i = imagpart
#...
#>>> x = Complex(3.0, -4.5)
#>>> x.r, x.i
#(3.0, -4.5)
#////////////////////////////////////////////////

class Room:

    def __init__(self, mC=300, K=20, Q_AC_Max = 1500, simulation_time = 12*60*60, control_step = 300):
        """
        All units SI units.
        All times are in seconds.

        """
        self.timestep = control_step #sec
        self.max_iteration = int(simulation_time/self.timestep)
        self.Q_AC_Max = Q_AC_Max
        self.mC = mC # kg.kj/(kg.degC)
        self.K = K # includes KA/thickness units--> W/(m.degC)
        #self.delta_T = self.T_in - self.T_out
        #self.reset()
        #self.schedule()

        plt.close()
        self.fig, self.ax = plt.subplots(1, 1)

    def reset(self,T_in = 20):
        self.iteration = 0 
        self.schedule()
        self.T_in = T_in


    def schedule(self):
        self.T_set = 25
        self.T_out = np.empty(self.max_iteration)
        self.T_out[:int(self.max_iteration/2)] = 28
        self.T_out[int(self.max_iteration/2):int(self.max_iteration)]= 32
        # Python broadcasting the non array type T_in 

    def update_Tin(self, action):
        self.Q_AC = action*self.Q_AC_Max # 
        self.T_in = self.T_in - 0.001*(self.timestep / self.mC) * (self.K*(self.T_in-self.T_out[self.iteration])+self.Q_AC)
        self.iteration +=1
        #return self.T_in

    
def p_controller(error):
    p_gain=0.025
    return p_gain*error

def I_controller(errorI):
    I_gain=0.02
    return I_gain*errorI

if __name__ == "__main__":

    x = Room()
    x.reset(T_in = np.random.randint(20,40))
    n_iter = 500
    Tin = np.empty(n_iter)
    Tset = (x.T_set) * np.ones_like(Tin)
    T_set_hi = Tset + 0.5
    T_set_lo = Tset - 0.5
    t = np.empty(n_iter)
    errorI = 0

    for i in range (n_iter):
        error = x.T_in - x.T_set
        errorI += error
        control_signal = p_controller(error) + I_controller(errorI)
        
        if abs(error) > 0.5:
            current_action = control_signal  # choose a power proportional to the gain
        else:
            current_action = 0
            
        t[i] = x.timestep/60 * i
        Tin[i] = x.T_in
        x.update_Tin(action=current_action)
        if x.iteration >=100:
            #print('game over : new espisode start')
            #time.sleep(1)
            x.reset(T_in = np.random.randint(20,40))    
        #print(current_action, x.T_in)
    plt.plot(t, Tin, 'b--', label='T_in')
    plt.plot(t, Tset, 'g--', label='T_set')
    plt.plot(t, T_set_hi, 'k--', label='T_set_hi')
    plt.plot(t, T_set_lo, 'k--', label='T_set_lo')
    plt.xlabel('Iteration time (min)')
    plt.ylabel('Temperature (deg. C)')
    plt.legend()
    plt.show()










