import numpy as np
import math as m
from scipy.optimize import minimize

#constantes; por el momento definimos todo a uno hasta consultar los valores reales
beta = 1
p = 1
lambd = 1
delta = 1
sigma = 1
rho = 1
epsilon_A = 1
epsilon_I = 1
gamma_A = 1
gamma_E = 1
gamma_I = 1
gamma_D = 1
d_D = 1
d_I = 1
theta = 1

#Definimos un vector con los parametros iniciales de cada lugar
# y_0 = (S_0, Q_0, E_0, A_0, I_0, D_0, R_0) 
y_0 = np.zeros(7)

#Definimos el vector t a usar
t = np.arange(30)
n = len(t)

# la funcion define las ecuacioes diferenciales presentados en la parte 2.1
# ----arXiv:2003.02985v1 [q-bio.PE] 6 Mar 2020
def fun(t,y_t,beta,epsilon_A):
    [S,Q,E,A,I,D,R] = y_t
    y = np.zeros(7)
    y[0] = -beta*S*(I + theta*A) - p*S + lambd*Q
    y[1] = p*S - lambd*Q
    y[2] = beta*S*(I + theta*A) - sigma*E
    y[3] = sigma*(1 - rho)*E - epsilon_A*A - gamma_A*A
    y[4] = sigma*(rho)*E - gamma_I*I - d_I*I - epsilon_I*I 
    y[5] = epsilon_A*A + epsilon_I*I - d_D*D - gamma_D*D
    y[6] = gamma_A*A + gamma_I*I + gamma_D*D    
    return y

#la función rungeKutta nos da una matriz donde cada fila está los datos del vector y, "y= (S, Q, E, A, I, D, R)" 
#El  numero de fila corresponde al numero ordinal de la madicón temporal de los datos
def rungeKutta(h, t, y_0, beta,epsilon_A):
    y = np.zeros((n,7))
    y[0,:] = y_0
    for i in range(1,n):
        k1 = h * fun(t[i-1], y[i-1,:], beta, epsilon_A)
        k2 = h * fun(t[i-1] + (h/2), y[i-1,:] + (k1/2), beta, epsilon_A)
        k3 = h * fun(t[i-1] + (h/2), y[i-1,:] + (k2/2), beta, epsilon_A)
        k4 = h * fun(t[i-1] + h, y[i-1,:] + k3, beta, epsilon_A)
        pendiente = (k1 + 2*k2 + 2*k3 + k4)/6
        y[i,:] = y[i-1,:] + pendiente   
    return y
#Definimos una funcion objetivo a minimizar, ecuación 2.4, se utilia los mis parametros
# ----arXiv:2003.02985v1 [q-bio.PE] 6 Mar 2020
def FunObj(x):
    [beta1, epsilon_A1] = x
    T = 0
    h = 0.01
    y_A = rungeKutta(h, t, y_0, beta1,epsilon_A1)
    y_R = np.ones((len(t),7)) #Este vector corresponderá a los datos reales
    for i in range(n):
        for j in range(7):
            T = T + abs(y_R[i,j] - y_A[i,j])
    return T
            
A = minimize(FunObj,[1,1],method='nelder-mead')  
#Prueba
print(A)

