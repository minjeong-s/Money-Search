# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Setting

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
from scipy.integrate import solve_ivp 
from sympy import *
from sympy.solvers import solve


# ## Functions

def u(y):
    u = ((y+b)**(1-eta) - b**(1-eta))/(1-eta)
    return u


def u_prime(y):
    u = 1/(y+b)**eta
    return u


# +
# when Adot = 0, steady-state에서, π가 주어졌을 때, A값을 찾아주는 함수
# (22)식 사용하면 된다

def find_steadystate_A(pi):
    Y = Symbol('Y')
    sol = solve( alpha*theta*(u_prime(Y)-1)/((1-theta)*u_prime(Y)+theta) - rho - pi, Y)
    
    if len(sol) == 2:
        Y_stat = sol[1]
    elif len(sol) == 1:
        Y_stat = sol[0]
    
    A_stat = (1-theta)*u(Y_stat) + theta*Y_stat
    
    print(f'steady_state A = {A_stat}, steady_state Y = {Y_stat}')
    return float(A_stat)


# -

def y_A(A):
    if A < (1-theta)*u(y_star) + theta*y_star:
        y = Symbol('y')
        sol = solve((1-theta)*u(y) + theta*y - A, y)
#         print(sol) 
        if len(sol) == 1:
            y_sol = sol[0]
        elif len(sol) == 2:
            y_sol = sol[1]
    
    else:
        y_sol = y_star
    
    return float(y_sol)


# T- 까지
def y_phi(phi):
    A = phi*M_0
    if A < (1-theta)*u(y_star) + theta*y_star:
        y = Symbol('y')
        sol = solve((1-theta)*u(y) + theta*y - A, y)
#         print(sol)
        
        if len(sol) == 1:
            y_sol = sol[0]
        elif len(sol) == 2:
            y_sol = sol[1]
    else:
        y_sol = y_star
        
    return float(y_sol)


# +
# 우리는 T에서 M이 한번 증가하기 전의 t(<T)에서 variable이 어떻게 변화하는지 보려고 하는 것이므로 
# t < T에서는 아직 M이 증가하지 않았으니까 M_0 그대로!

def y_phi_forODE(phi):
    A = phi*M_0
    if A < (1-theta)*u(y_star) + theta*y_star:
        y = Symbol('y')
        sol = solve((1-theta)*u(y) + theta*y - A, y)
#         print(sol)
        
        if len(sol) == 1:
            y_sol = sol[0][0]
        elif len(sol) == 2:
            y_sol = sol[1][0]
    else:
        y_sol = y_star
        
    return float(y_sol)


# -

# ## Parameters

# +
# parameters related to the utility function
b = 0.01
eta = 0.5
rho = 0.04
theta = 0.5 # buyer's bargaining power
alpha = 0.5

pi_0 = 0
As_0 = find_steadystate_A(pi_0)
# -

# π=0으로 고정. As_0 고정
# ϕs_0 구하기, A = ϕM 사용해서 구하면 됨!!
M_0 = 1
mu = 0.1
M_1 = M_0*(1+mu)
phi_s_0 = As_0/M_0
phi_s_0

pi_mu = mu
As_mu = find_steadystate_A(pi_mu)

As_0/(1+mu)

# +
y = Symbol('y')
f = ((y+b)**(1-eta) - b**(1-eta))/(1-eta)
f.diff(y)

y_sol = solve(f.diff(y)-1, y)

if len(y_sol) == 2:
    y_star = y_sol[1]

elif len(y_sol) == 1:
    y_star = y_sol[0]
# -

# ## Solving ODE

# phi에 대한 ODE (28-2)식
G = lambda t,phi: phi* ( -rho + alpha*theta*(  ( u_prime(y_phi_forODE(phi))-1 ) / ( (1-theta)*u_prime(y_phi_forODE(phi)) + theta )  ))

T = 10
sol = solve_ivp(fun = G, t_span = [0,T], y0 = [phi_s_0/(1+mu)], method = 'RK45', t_eval = np.arange(0, T, T/1000))

# ### Evolution of A

# +
# A에 대한 그래프를 그릴거니까 phi에 대해서 푼 ODE에 *M_0을 해줘야함 
# t < T에서의 phi, A 변화니까 M_0을 곱해줘야함 
# t = T 일 때만 M_1을 곱해서 A가 다시 어디로 가는지를 봐야함

phi_t = sol.y[0]
A_t = np.zeros(len(sol.y[0]))
A_t[1:] = sol.y[0][1:]*M_0
A_t[0] = sol.y[0][0]*M_1
# -

plt.plot(sol.t, A_t)
plt.xlabel(f'tau = {T}-t')
plt.ylabel(f'$A^{T}_t$')
plt.title('Reverse evolution of A')
plt.grid(True)
plt.axhline(y = As_0, color='gray', linestyle='--', label = 'As_0')
plt.plot(0, A_t[0], marker='o')
# for all t < T, dotA < 0, proposition 4와 일치

# t = 0에서의 값: As_0에서 점프가 일어나는지? Yes 점프가 일어난다 
# 위에 그래프에서 회색 점선이 As_0값인데 tau = 1, 즉 t = 0 일때 점프가 일어난 것을 확인 가능
As_0 - A_t[-1]*M_0 

# +
# t = T에서 M이 M_1 = M*(1+mu)로 점프하면 As_0 값으로 점프하게 되는지?
# t = T일 때 M_1으로 바뀌면서 As_0으로 돌아가는 거니까 M_1을 곱해야함

A_t[0] - As_0 # As_0값으로 돌아간다
# -

# ### 증가율 체크

# +
# t=T일 때 mu를 따르는 new Adot/A 그래프가 그려지고, 얘를 따라서 As로 다시 back하는건지 증가율을 체크해서 확인해보기 

# +
Amin = 0
Amax = As_0*2
nA = int(Amax*10)
Agrid = np.linspace(Amin, Amax, nA)

Ygrid = np.zeros(nA)
Ygrid[Agrid >= (1-theta)*u(y_star) + theta*y_star] = y_star # by (19)

for idx, val in enumerate(Agrid[Agrid < (1-theta)*u(y_star) + theta*y_star]): # by (18)
    Y = Symbol('Y')
    Y_sol = solve((1-theta)*u(Y) + theta*Y - val, Y)
    if len(Y_sol) == 2:
        Ygrid[idx] = Y_sol[1]
    elif len(Y_sol) == 1:
        Ygrid[idx] = Y_sol[0]
#     print(idx)


Adot_over_A_0 = pi_0 + rho - alpha*theta*(u_prime(Ygrid)-1)/((1-theta)*u_prime(Ygrid)+theta)
Adot_over_A_mu = pi_mu + rho - alpha*theta*(u_prime(Ygrid)-1)/((1-theta)*u_prime(Ygrid)+theta)
# -

ATminus = sol.y[0][0]*M_0
ATminus

ATminus == As_0/(1+mu)

dA = ATminus*(pi_mu + rho - alpha*theta*(u_prime(y_A(ATminus))-1)/((1-theta)*u_prime(y_A(ATminus))+theta))

ATminus + dA

As_0

ATminus + dA == As_0 ##################


# ### Discrete하게 그려보기

def find_next_phi(phi):
    phi_next = phi + phi*(-rho+ alpha*theta*(u_prime(y_phi(phi))-1)/((1-theta)*u_prime(y_phi(phi))+theta))
    return phi_next


# +
T = 10

phi0 = phi_s_0/(1+mu )

phi_path = []
phi_path.append(phi0)

# 초기값 설정
i = 0
phi = phi0

while i < T:
    i += 1
    phi_new = find_next_phi(phi)
    phi_path.append(phi_new)
    
    # update
    phi = phi_new
    
phi_path = np.array(phi_path)
A_path = phi_path*M_0 

# T- 까지의 A값을 구하는 것이므로
# -

A_path

dA = A_path[0]*(pi_mu + rho - alpha*theta*(u_prime(y_A(A_path[0]))-1)/((1-theta)*u_prime(y_A(A_path[0]))+theta))

A_path[0] + dA

As_0 == A_path[0] + dA ######################

# ### $\dot{A}/A$ 그리기
# - Fig 5.

# initial jump
plt.plot(Agrid, Adot_over_A_0)
plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.plot(As_0,0, marker='o', label = 'As_0')
plt.plot(A_t[-1],0, marker='o', label = 'A0')
plt.legend(loc=4)
plt.title('Phase diagram $\dot{A}/A$: initial jump')

# +
# At t = T, come back to initial steady-state

plt.plot(Agrid, Adot_over_A_0)
plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.plot(As_0,0, marker='o', label = 'As_0')
plt.plot(A_t[-1],0, marker='o', label = 'A0')
plt.plot(A_t[0],0, marker='o', label = '$A^T$')
# plt.xlim((4.8, 4.9))
plt.legend(loc=4)
plt.title('Phase diagram $\dot{A}/A$: end point coming back to initial steady-state')
# -

plt.plot(Agrid, Adot_over_A_0)
plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.plot(As_0,0, marker='o', label = 'As_0')
for i in range(len(A_t)):
    plt.plot(A_t[i], 0, marker = 'o')   
plt.title('Phase diagram $\dot{A}/A$: Moving')
plt.xlim((1, 1.2))
plt.axvline(As_0/(1+mu), color='gray', linestyle='--', label = '$A^s/(1+\mu)$')
plt.legend(loc=4)


plt.plot(Agrid, Adot_over_A_0)
plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.plot(As_0,0, marker='o', label = 'As_0')
for i in range(len(A_t)):
    plt.plot(A_t[i], 0, marker = 'o')   
plt.legend(loc=4)
plt.title('Phase diagram $\dot{A}/A$')
# plt.xlim((4.8, 4.9))


