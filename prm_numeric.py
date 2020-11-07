###################################################
# Here we solve the Rinzel-Pinsky model from the  #
# 1994 paper using numerical methods. This model  #
# is a 2 compartment, 8 variable model. The two   #
# compartments consist of a slow and fast method  #
# which are the sodium spiking and slow calcium   #
# channels. We simulate this with a single neuron #
# first and then proceed to connect multiple      #
# elements together to form more complex systems. #
#                                                 #
# The breakdown of the following project is such: #
# -Numeric solution to the single element PR      #
#  model.                                         #
# -Using this single element, we build a more     #
#  complex system which connects these elements.  #
# -Finally, we apply our own modification to the  #
#  PR model to take into account proximal signal  #
#  time to investigate effects of finite signal   #
#  travel speeds.                                 #
#                                                 #
#Author: Elliot Kisiel, Biswajit Sahoo,           #
#        Chris Khoury                             #
#                                                 #
#Change Log:                                      #
#	Created November 5th                      # 
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
#                                                 #
###################################################

import numpy as np
import matplotlib.pyplot as plt
import math as m

p = 0.5
Cm = 3 #uF/cm^2
gc = 2.1
gL = 0.1
gNa = 30
gDR = 15
gCa = 10
gAHP = 0.8
gC = 15
E_L = -60 #mV
E_Na = 60 #mV
E_K = -75 #mV
E_Ca = 80 #mV

#initial values:
t_start = 0 #ms
t_end = 5000 #ms
N = 200000
step = (t_end - t_start)/N
t = np.arange(t_start,t_end,step)
Vd = np.zeros(N)
Vs = np.zeros(N)
h = np.zeros(N)
n = np.zeros(N)
s = np.zeros(N)
c = np.zeros(N)
q = np.zeros(N)
Ca = np.zeros(N)

Vd[0] = -65
Vs[0] = -65
h[0] = 0.5
n[0] = 0.5
s[0] = 0.5
c[0] = 0.5
q[0] = 0.5
Ca[0] = 0.5

#n' = a_n(1-n) - b_n(n)

#m variables
def am(V):
	return (-0.32 * (V + 46.9)/(m.exp(-1*(V + 46.9)/4) - 1))
def bm(V):
	return ( 0.28 * (V + 19.9)/(m.exp((V + 19.9)/5) - 1))

#h variables
def ah(V):
	return (0.128 * m.exp((-43 - V)/18))
	
def bh(V):
	return (4 / (1 + m.exp(-1 * (V + 20) / 5)))

#n variables
def an(V):
	return (-0.32 * (V + 46.9)/(m.exp(-1*(V + 46.9)/4) - 1))
	
def bn(V):
	return (0.25 * m.exp(-1 * (V +  40) / 40))


#s variables
def a_s(V):
	return (1.6 / (1 + m.exp(-0.072 * (V - 5))))
def bs(V):
	return (0.02 * (V + 8.9)/(m.exp((V + 8.9)/5) - 1))


#c variables
def ac(V):
	if(V <= -10):
		return (0.0527 * (m.exp((V + 50)/11) - m.exp((V + 53.5)/27)))
	else:
		return (2 * m.exp(-1 * (V + 53.5)/27))
def bc(V):
	if(V <= -10):
		return ((2 * m.exp(-1 * (V + 53.5)/27)) - (0.0527 * (m.exp((V + 50)/11) - m.exp((V + 53.5)/27))))
	else:
		return 0
	
def XCa(Ca):
	return min(Ca/250, 1)

#q variables 
def aq(Ca):
	return min(0.00002*Ca, 0.01)

bq = 0.001


#calcium variables
def minf(Vd):
	return (am(Vd)/(am(Vd) + bm(Vd)))

def h_diffequ(Vs,h):
	return (ah(Vs) * (1-h) - bh(Vs) * h)

def n_diffequ(Vs,n):
	return (an(Vs) * (1-n) - bn(Vs) * n)

def s_diffequ(Vd,s):
	return (a_s(Vd) * (1-s) - bs(Vd) * s)

def c_diffequ(Vd,c):
	return (ac(Vd) * (1-c) - bc(Vd) * c)

def q_diffequ(Ca,q):
	return (aq(Ca) * (1-q) - bq * q)

def gNa_full(Vs,h):
	return (gNa * minf(Vs) * h)

def gDR_full(n):
	return (gDR * n)

def gC_full(c, Ca):
	return (gC * c * XCa(Ca))

def gCa_full(s):
	return (gCa * s**2)

def gAHP_full(q):
	return (gAHP * q)

def Ca_diffequ(Ca, Vd, s):
	return (-0.13 * gCa_full(s) * (Vd - E_Ca) - 0.075 * Ca)

def Vs_diffequ(Vs,h,n,Vd,Is):
	return	(-1 * gL * (Vs - E_L) - gNa_full(Vs,h) * (Vs - E_Na) - gDR_full(n) * (Vs - E_K) + gc*(Vd - Vs) / p + Is / p)/Cm

def Vd_diffequ(Vd, s, q, c, Ca, Vs, Isyn):
	return	(-1 * gL * (Vd - E_L) - gCa_full(s) * (Vd - E_Ca) - gAHP_full(q) * (Vd - E_K) - gC_full(c, Ca) * (Vd - E_K) + gc*(Vs - Vd) /(1 - p) + Isyn /(1 - p))/Cm

def rk4_integrate_voltage(step, Vs, Vd, h, n, s, c, q, Ca, i, Is = 0, Isyn = 0):
	K1Vs = Vs_diffequ(Vs[i],h[i],n[i],Vd[i],Is)
	K1Vd = Vd_diffequ(Vd[i], s[i], q[i], c[i], Ca[i], Vs[i], Isyn)
	K1h = h_diffequ(Vs[i],h[i])
	K1n = n_diffequ(Vs[i],n[i])
	K1s = s_diffequ(Vd[i],s[i])
	K1c = c_diffequ(Vd[i],c[i])
	K1q = q_diffequ(Vd[i],q[i])
	K1Ca = Ca_diffequ(Ca[i], Vd[i],s[i])

	K2Vs = Vs_diffequ(Vs[i] + (step * K1Vs / 2),h[i]+ (step * K1h / 2),n[i]+ (step * K1n / 2),Vd[i]+ (step * K1Vd / 2),Is)
	K2Vd = Vd_diffequ(Vd[i]+ (step * K1Vd / 2), s[i]+ (step * K1s / 2), q[i]+ (step * K1q / 2), c[i]+ (step * K1c / 2), Ca[i]+ (step * K1Ca / 2), Vs[i]+ (step * K1Vs / 2), Isyn)
	K2h = h_diffequ(Vs[i]+ (step * K1Vs / 2),h[i]+ (step * K1h / 2))
	K2n = n_diffequ(Vs[i]+ (step * K1Vs / 2),n[i]+ (step * K1n / 2))
	K2s = s_diffequ(Vd[i]+ (step * K1Vd / 2),s[i]+ (step * K1s / 2))
	K2c = c_diffequ(Vd[i]+ (step * K1Vd / 2),c[i]+ (step * K1c / 2))
	K2q = q_diffequ(Vd[i]+ (step * K1Vd / 2),q[i]+ (step * K1q / 2))
	K2Ca = Ca_diffequ(Ca[i]+ (step * K1Ca / 2), Vd[i]+ (step * K1Vd / 2),s[i]+ (step * K1s / 2))

	K3Vs = Vs_diffequ(Vs[i] + (step * K2Vs / 2),h[i]+ (step * K2h / 2),n[i]+ (step * K2n / 2),Vd[i]+ (step * K2Vd / 2),Is)
	K3Vd = Vd_diffequ(Vd[i]+ (step * K2Vd / 2), s[i]+ (step * K2s / 2), q[i]+ (step * K2q / 2), c[i]+ (step * K2c / 2), Ca[i]+ (step * K2Ca / 2), Vs[i]+ (step * K2Vs / 2), Isyn)
	K3h = h_diffequ(Vs[i]+ (step * K2Vs / 2),h[i]+ (step * K2h / 2))
	K3n = n_diffequ(Vs[i]+ (step * K2Vs / 2),n[i]+ (step * K2n / 2))
	K3s = s_diffequ(Vd[i]+ (step * K2Vd / 2),s[i]+ (step * K2s / 2))
	K3c = c_diffequ(Vd[i]+ (step * K2Vd / 2),c[i]+ (step * K2c / 2))
	K3q = q_diffequ(Vd[i]+ (step * K2Vd / 2),q[i]+ (step * K2q / 2))
	K3Ca = Ca_diffequ(Ca[i]+ (step * K2Ca / 2), Vd[i]+ (step * K2Vd / 2),s[i]+ (step * K2s / 2))

	K4Vs = Vs_diffequ(Vs[i] + (step * K3Vs),h[i]+ (step * K3h),n[i]+ (step * K3n),Vd[i]+ (step * K3Vd),Is)
	K4Vd = Vd_diffequ(Vd[i]+ (step * K3Vd), s[i]+ (step * K3s), q[i]+ (step * K3q), c[i]+ (step * K3c), Ca[i]+ (step * K3Ca), Vs[i]+ (step * K3Vs), Isyn)
	K4h = h_diffequ(Vs[i]+ (step * K3Vs),h[i]+ (step * K3h))
	K4n = n_diffequ(Vs[i]+ (step * K3Vs),n[i]+ (step * K3n))
	K4s = s_diffequ(Vd[i]+ (step * K3Vd),s[i]+ (step * K3s))
	K4c = c_diffequ(Vd[i]+ (step * K3Vd),c[i]+ (step * K3c))
	K4q = q_diffequ(Vd[i]+ (step * K3Vd),q[i]+ (step * K3q))
	K4Ca = Ca_diffequ(Ca[i]+ (step * K3Ca), Vd[i]+ (step * K3Vd),s[i]+ (step * K3s))

	Vs[i+1] = Vs[i] + step * (K1Vs + 2 * K2Vs + 2 * K3Vs + K4Vs) / 6
	Vd[i+1] = Vd[i] + step * (K1Vd + 2 * K2Vd + 2 * K3Vd + K4Vd) / 6
	h[i+1] = h[i] + step * (K1h + 2 * K2h + 2 * K3h + K4h) / 6
	n[i+1] = n[i] + step * (K1n + 2 * K2n + 2 * K3n + K4n) / 6
	s[i+1] = s[i] + step * (K1s + 2 * K2s + 2 * K3s + K4s) / 6
	c[i+1] = c[i] + step * (K1c + 2 * K2c + 2 * K3c + K4c) / 6
	q[i+1] = q[i] + step * (K1q + 2 * K2q + 2 * K3q + K4q) / 6
	Ca[i+1] = Ca[i] + step * (K1Ca + 2 * K2Ca + 2 * K3Ca + K4Ca) / 6



for i in range(N-1):
	rk4_integrate_voltage(step,Vs,Vd,h,n,s,c,q,Ca,i)

plt.plot(t,Vd)
plt.show()

