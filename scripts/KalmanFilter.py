# -*- coding: utf-8 -*-
"""
Created on Sun Apr 05 18:51:36 2015

Kalman filter class for discrete time linear systems
    xkp1 = Ak*xk + Bk*uk + vk
    yk = Ck*xk + wk

    Pk: cov matrix of estimation error
    Rk: cov matrix of measurement noise wk
    Qk: cov matrix of state noise vk

@author: S. Bertrand
"""

import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    
    def __init__(self, dimState=0, dimInput=0, dimOutput=0):
        # state, input and measurement dimensions
        self.dimState = dimState        
        self.dimInput = dimInput
        self.dimOutput = dimOutput

        # system matrices
        self.Ak = np.zeros((self.dimState, self.dimState))
        self.Bk = np.zeros((self.dimState, self.dimInput))
        self.Ck = np.zeros((self.dimOutput, self.dimState))
        
        # state, input and measurement vectors
        self.xk = np.zeros((self.dimState,1))
        self.uk = np.zeros((self.dimInput,1))
        self.yk = np.zeros((self.dimOutput,1))

        # state, state noise and measurement noise covariance matrices
        self.Pk = np.zeros((self.dimState, self.dimState)) 
        self.Qk = np.zeros((self.dimState, self.dimState)) 
        self.Rk = np.zeros((self.dimOutput, self.dimOutput))
        
        
    # display function in command line
    def __repr__(self):
        message = "Kalman Filter: \n"
        message += " - dimensions: state: {}, input: {}, output: {}\n".format(self.dimState, self.dimInput, self.dimOutput)
        message += " - Ak={}\n - Bk={}\n - Ck={}".format(self.Ak, self.Bk, self.Ck)        
        return message

    # display funtion with print
    def __str__(self):        
        message = "Kalman Filter: \n"
        message += " - dimensions: state: {}, input: {}, output: {}".format(self.dimState, self.dimInput, self.dimOutput)
        message += " - Ak={}\n - Bk={}\n - Ck={}".format(self.Ak, self.Bk, self.Ck)        
        return message


    def setAk(self, Ak):
        self.Ak = Ak        

    def setBk(self, Bk):
        self.Bk = Bk        

    def setCk(self, Ck):
        self.Ck = Ck        

    def setStateEquation(self, Ak, Bk):
        self.Ak = Ak
        self.Bk = Bk

    # set a priori estimate and cov matrix        
    def initFilter(self, x0, P0):
        self.xk = x0
        self.Pk = P0
        
    def setRk(self, Rk):
        self.Rk = Rk
        
    def setQk(self, Qk):
        self.Qk = Qk

    # prediction step of the Kalman filter, with input uk
    def predict(self, uk):
        self.uk = uk
        # state prediction
        self.xk = np.dot(self.Ak, self.xk) + np.dot(self.Bk, self.uk) 
        # covariance prediction
        self.Pk = np.dot( self.Ak,  np.dot( self.Pk, self.Ak.T ) ) + self.Qk
        #print "predict\n"

    # update step of the Kalman filter, with measurement
    def update(self, yk):
        self.yk = yk
        # innovation
        innovk = self.yk - np.dot(self.Ck, self.xk)
        # innovation covariance
        Sk = np.dot(self.Ck, np.dot(self.Pk , self.Ck.T) ) + self.Rk
        # Kalman gain
        Kk = np.dot(self.Pk, np.dot(self.Ck.T , np.linalg.inv(Sk) )  )
        # state update
        self.xk = self.xk + np.dot(Kk , innovk)
        # covariance update
        self.Pk = self.Pk - np.dot(Kk , np.dot(self.Ck,self.Pk) )
        #print "update\n"
        
        
    # prediction step and update step
    def runOneStep(self, uk, yk):
        self.oneStepPredict(uk)
        self.update(yk)        
        
        
        
        

if __name__=='__main__':
    
    # ***********************************
    # validation example:
    # ***********************************    
    #   system:
    #   x1(k+1) = x1(k) + x2(k) + v1(k)
    #   x2(k+1) = x2(k) + u(k)  + v2(k)
    #   y(k) = x1(k) + w(k)
    #
    #   noise characteristics:
    #   v1(k) ~ N(0,0)
    #   v2(k) ~ N(0,1)
    #   w(k)  ~ N(0,2)
    #
    #   filter initialisation :
    #   x1hat(0) = 0
    #   x2hat(0) = 10
    #   P(0)  = [ 2     0 ]
    #           [ 0     3 ]
    #
    # ***********************************       
    
    
    nx = 2
    nu = 1
    ny = 1    
    Ak = np.array( [[1.0,1.0],[0.0,1.0]] )
    Bk = np.array( [ [0.0],[1.0] ] )
    Ck = np.array( [[1.0, 0.0]] )
    Qk = np.array( [ [0.0, 0.0], [0.0, 1.0] ] )
    Rk = np.array([2.0])
    x0hat = np.array([ [0.0],[10.0] ])
    P0 = np.array( [ [2.0,0.0],[0.0,3.0] ] )  
    
    kf = KalmanFilter(nx, nu, ny)
    kf.setStateEquation(Ak, Bk)
    kf.setCk(Ck)
    kf.setQk(Qk)
    kf.setRk(Rk)
    kf.initFilter( x0hat , P0 )
    
    
    # time history of measurement yk (no value defined for k=0)
    Y = [ [], 9.0, 19.5, 29]
    # time history of input uk (no value defined for k=0)
    U = [ [], 0.0, 0.0, 0.0]

    
    # time history for display
    nbPts = 4
    X = np.zeros((nx, nbPts))
    X[:,0] = x0hat[:,0]
    indexesK = range(0,4)
    sigmaX = np.zeros((2, nbPts))
    sigmaX[0,0] = np.sqrt(P0[0,0])
    sigmaX[1,0] = np.sqrt(P0[1,1])    
    
    # simulation loop
    for k in range(1,4):
        
        uk = U[k]
        yk = Y[k]
        print("\n- k={},   uk={},   yk={}".format(k, uk, yk))
        
        # prediction step
        kf.predict(uk)
        print("-- after prediction step:")
        print("     xk={}".format(kf.xk))
        print("     Pk={}".format(kf.Pk))
        
        # update step
        kf.update(yk)
        print("-- after update step:")
        print("     xk={}".format(kf.xk))
        print("     Pk={}".format(kf.Pk))
        
        X[:,k] = kf.xk[:,0]
        sigmaX[0,k] = np.sqrt(kf.Pk[0,0])
        sigmaX[1,k] = np.sqrt(kf.Pk[1,1])
    
    
    
    # results to be found (after update)
    Xsolution = np.array([[0, 9.28571429, 19.33636364, 29.05405405],
                          [10, 9.57142857,  9.86363636,  9.78255528 ]])
    sigmaXsolution = np.array([[1.414213562373095145e+00,	1.195228609334393788e+00,	1.221027882936786657e+00,	1.208080899385243878e+00],
                               [1.732050807568877193e+00,	1.647508942095827988e+00,	1.445997610962442392e+00,	1.369194242864347766e+00]])


    plt.close('all')

    # plot solution and computed values
    plt.figure()    
    l1, = plt.plot(indexesK, X[0,:], 'b*', markersize=10)
    plt.plot(indexesK, X[0,:]+3*sigmaX[0,:], 'k*', markersize=10)
    plt.plot(indexesK, X[0,:]-3*sigmaX[0,:], 'k*', markersize=10)    
    l2, = plt.plot(indexesK, Xsolution[0,:], 'b')
    plt.plot(indexesK, Xsolution[0,:]+3*sigmaXsolution[0,:], 'k')
    plt.plot(indexesK, Xsolution[0,:]-3*sigmaXsolution[0,:], 'k')
    plt.xlabel('k')
    plt.ylabel('x1(k)')
    plt.grid(True)
    plt.legend([l1,l2], ['computed', 'solution'], loc=2)
    plt.xlim([-0.5, 3.5])
    plt.title('Estimate (blue) and 3-sigma incertitude (black)')
    
    
    plt.figure()
    l1, = plt.plot(indexesK, X[1,:], 'r*', markersize = 10)
    plt.plot(indexesK, X[1,:]+3*sigmaX[1,:], 'k*', markersize=10)
    plt.plot(indexesK, X[1,:]-3*sigmaX[1,:], 'k*', markersize=10)  
    l2, = plt.plot(indexesK, Xsolution[1,:], 'r')
    plt.plot(indexesK, Xsolution[1,:]+3*sigmaXsolution[1,:], 'k')
    plt.plot(indexesK, Xsolution[1,:]-3*sigmaXsolution[1,:], 'k')
    plt.xlabel('k')
    plt.ylabel('x2(k)')
    plt.grid(True)
    plt.legend([l1,l2], ['computed', 'solution'], loc=1)
    plt.xlim([-0.5, 3.5])
    plt.title('Estimate (red) and 3-sigma incertitude (black)')
    