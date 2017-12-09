# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 20:31:54 2017

@author: mlopes
"""
import numpy as np


def Q2pol(Q, eta = 5):
	pol = np.zeros(Q.shape)
	maximo = Q[0][0]

	for i in range(0, Q.shape[0]):
		for j in range(0, Q.shape[1]):
			if maximo < Q[i][j]:
				maximo = Q[i][j]
				index = j
		pol[i][index] = 1
		maximo = Q[i][0]

	return pol




def Q2V(Q):
	V = [0] * Q.shape[0]
	
	for i in range(0,Q.shape[0]):
		V[i] = max(Q[i])

	return V



def QhasError(prevQ, currQ):
	prevV = Q2V(prevQ)
	currV = Q2V(currQ)

	return not np.array_equal(prevV, currV)

	

class myRL:

	def __init__(self, nS, nA, gamma):
		self.nS = nS
		self.nA = nA
		self.gamma = gamma
		self.Q = np.zeros((nS,nA))	#no estados / no accoes

		
	def traces2Q(self, trace):	#trace eh matriz / self eh o gajo em cima
		currQ = (self.Q).copy()
		prevQ = (self.Q).copy()
		alpha = 0.1
		error = True

		while(error):

			for step in trace: # por cada tupulo de step
				estadoActual = int(step[0])
				accao        = int(step[1])
				estadoSeg    = int(step[2])
				recompensa   = step[3]

				# Q(x,a) = Q(x,a)*(1-alfa) + alfa * (recompensa + gama( Max Q(y,b) ) )
				currQ[estadoActual][accao] =  (1 - alpha) * currQ[estadoActual][accao] + alpha * (recompensa + self.gamma * (max(currQ[estadoSeg])))

			error = QhasError(prevQ = prevQ, currQ = currQ)
			prevQ = currQ.copy()

		self.Q = currQ.copy()

		return self.Q
