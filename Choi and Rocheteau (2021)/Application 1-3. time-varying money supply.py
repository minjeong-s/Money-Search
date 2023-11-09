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


# +
# using (18), (19), define function y of A
# 왜냐면 (28-1) 식을 보면, 오른쪽 텀을 A에 대한 식으로 만드는게 좋으니까

def y_A_forODE(A):
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


# ## Parameters

# parameters related to the utility function
b = 0.01
eta = 0.5
rho = 0.04
theta = 0.5 # buyer's bargaining power
alpha = 0.5

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

# # Inflationary or Disinflationary policy

# ## Parameters

# +
# T의 절대적인 숫자보다 0에서 T까지를 몇개의 구간으로 나눴는지가 중요함. 
# 그 구간의 갯수를 나중에 ODE 풀 때 t_span 인자의 끝점으로 전달해주기
# tdot = 1

policy = 'inflationary' # Choose btw "inflationary / disinflationary"
pi0 = 0.2 # 시작 값

gridnum = 20 # 정수로 넣어주기

t = np.linspace(0,gridnum-1,gridnum)

T = int(np.ceil(gridnum*0.8))
print(f'T = {T}')


# -

# ## Pi functions

# +
# pi_t를 t에 대한 함수로 만들기, 나중에 ODE에 넣어주기 위함
def pi_t(t,policy,  pi0):
    if policy == 'disinflationary':
        if t < T:
            pi_t = pi0 - (pi0/gridnum)*t
        else:
            pi_t = pi0 - (pi0/gridnum)*T
            
    if policy == 'inflationary':
        if t < T:
            pi_t = pi0 + (pi0/gridnum)*t
        else:
            pi_t = pi0 + (pi0/gridnum)*T
    return pi_t

pi_t = np.vectorize(pi_t)


# +
# tau = T - t
def pi_tau(tau, policy,  pi0):
    t = T - tau
    if policy == 'disinflationary':
        if t < T:
            pi = pi0 - (pi0/gridnum)*t
        else:
            pi = pi0 - (pi0/gridnum)*T
    if policy == 'inflationary':
        if t < T:
            pi = pi0 + (pi0/gridnum)*t
        else:
            pi = pi0 + (pi0/gridnum)*T
    return pi

pi_tau = np.vectorize(pi_tau)
# -

# check: True
(pi_t(T,policy,  pi0) < alpha*theta/(1-theta) - rho) & (pi_t(T,policy,  pi0)>= -rho)

plt.plot(t, pi_t(t,policy,  pi0))
plt.xlabel('t')
plt.ylabel('$\pi_t$')
plt.title('$\pi_t$')
plt.grid(True)

tau = T - t
plt.plot(tau, pi_tau(tau,policy,  pi0))
plt.grid(True)
plt.xlabel('tau = T - t')
plt.ylabel('$\pi_{T-t}$')
plt.title('$\pi_{T-t}$')

# ## Adot = 0 locus

# +
# Adot = 0 일 때, 각 pi 값에 따른 As의 값을 구하기

Ygrid = np.zeros(len(t))

for idx, val in enumerate(pi_t(t,policy,  pi0)):
#     print(idx)
    Y = Symbol('Y')
    Y_sol = solve( alpha*theta*(u_prime(Y)-1)/((1-theta)*u_prime(Y)+theta) - rho - val, Y)
    if len(Y_sol) == 1:
        Ygrid[idx] = Y_sol[0]
    elif len(Y_sol) == 2:
        Ygrid[idx] = Y_sol[1]
        
# (18), (19) 사용해서 y에서 a구하기
As_list = (1-theta)*u(Ygrid) + theta*Ygrid
# -

plt.plot(t, As_list, label = '$\dot{A}=0$')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('A')
plt.title('$\dot{A}=0$')
plt.legend()

As_list[T] # 끝값, T기에 도달하는 steady-state 값, As_list[-1] 이렇게 해도 상관 없음

# ## backward ODE로 A0찾기

F = lambda t,A: A * ( -rho - pi_tau(t,policy,  pi0) + alpha*theta*(  ( u_prime(y_A_forODE(A))-1 ) / ( (1-theta)*u_prime(y_A_forODE(A)) + theta )  )      ) 
sol = solve_ivp(fun = F, t_span = [0,T], y0 = [As_list[T]], method = 'RK45', t_eval = np.arange(0, T+1, 1)) 
# 그래프를 더 부드럽게 그리고 싶으면 t_eval에 추가해주기 

# 거꾸로 푼 것이기 때문에 그래프도 거꾸로 읽어야함
plt.plot(sol.t, sol.y[0])
plt.xlabel('T-t')
plt.ylabel('$A_{T-t}$')
plt.grid(True)
plt.title('$A_{T-t}$')

# +
plt.plot(np.flip(T-sol.t), np.flip(sol.y[0]), label = '$A_t$')

plt.plot(t, As_list, label = '$\dot{A}=0$', color ='red')
plt.grid(True)
plt.title('Time path of $A_t$')
plt.xlabel('t')
plt.ylabel('$A_t$')
plt.legend()

# -

sol.t

# 시간순으로 바꿔줌 
A_t = np.flip(sol.y[0])
A_t


# ## Discrete 하게 구해보기

# tau version 
def find_next_A(A, pi): 
    A_next = A + A*(-rho - pi + alpha*theta*(  ( u_prime(y_A(A))-1 ) / ( (1-theta)*u_prime(y_A(A)) + theta )))
    return A_next


# +
A0 = As_list[T]

A_path = []
A_path.append(A0)

i = 0
A = A0

while i < T:
    pi = pi_tau(i, policy, pi0)
    Anew = find_next_A(A, pi)
    A_path.append(Anew)
    # update 
    A = Anew
    i += 1
# -

plt.plot(A_path, label = 'discrete')
plt.plot(sol.t, sol.y[0], label = 'continuous')
plt.grid(True)
plt.legend()

# ## 증가율 확인

# +
# 각 pi에 따른 Adot/A 그래프

y_A_vec = np.vectorize(y_A)
Adot_over_A_list = []
Amin = 0
Amax = 5
nA = 100
Agrid = np.linspace(Amin, Amax, nA)

for pi in pi_t(t, policy, pi0):
    Adot_over_A = rho + pi - alpha*theta*(u_prime(y_A_vec(Agrid))-1)/((1-theta)*u_prime(y_A_vec(Agrid))+theta)
    Adot_over_A_list.append(Adot_over_A)

# +
# A0랑 A1 두개만 그려본 것
plt.plot(Agrid, Adot_over_A_list[0], label = 'pi0')
plt.plot(Agrid, Adot_over_A_list[1], label = 'pi1')
plt.plot(Agrid, Adot_over_A_list[2], label = 'pi2')
plt.plot(Agrid, Adot_over_A_list[3], label = 'pi3')

plt.grid(True)

plt.plot(A_t[0], 0, marker='o', label = 'A0')
plt.plot(A_t[1], 0, marker='o', label = 'A1')
plt.plot(A_t[2], 0, marker='o', label = 'A2')
plt.plot(A_t[3], 0, marker='o', label = 'A3')

plt.legend(loc=4)
plt.xlim((0,0.2))

# +
# 각 i기마다 A_i가 Adot_over_A(pi_i) 그래프를 따라서 A_(i+1)로 변하는지 확인 (증가율 확인)
# discrete하게 구한 값에 대하여

A_path_flip = np.flip(A_path) # 시간순으로 바꿔주기

for i in np.arange(0,T,1):
    dA = A_path_flip[i]*(rho + pi_t(i, policy, pi0) - alpha*theta*(u_prime(y_A_vec(A_path_flip[i]))-1)/((1-theta)*u_prime(y_A_vec(A_path_flip[i]))+theta))
    A_next = A_path_flip[i] + dA
    print(A_next - A_path_flip[i+1])
    

# +
# 각 i기마다 A_i가 Adot_over_A(pi_i) 그래프를 따라서 A_(i+1)로 변하는지 확인 (증가율 확인)
# continuous하게 구한 값에 대하여

for i in np.arange(0,T,1):
    dA = A_t[i]*(rho + pi_t(i, policy, pi0) - alpha*theta*(u_prime(y_A_vec(A_t[i]))-1)/((1-theta)*u_prime(y_A_vec(A_t[i]))+theta))
    A_next = A_t[i] + dA
    print(A_next - A_t[i+1])
    
    
# continuous로 구한게 오차가 더 작음. 절반정도
# -

# # Non-monotone policy

# ## Parameters

# +
# T의 절대적인 숫자보다 0에서 T까지를 몇개의 구간으로 나눴는지가 중요함. 
# 그 구간의 갯수를 나중에 ODE 풀 때 t_span 인자의 끝점으로 전달해주기
# tdot = 1

pi0 = 0.1 # T에 도달하는 값
pi_max = 0.3 

gridnum = 50 # 정수로 넣어주기

t = np.linspace(0,gridnum-1,gridnum)

T1 = int(np.ceil(gridnum*0.1)) # pi가 변하기 시작하는 때 
T2 = int(np.ceil(gridnum*0.8)) # T, pi가 변화를 멈추는 값

print(f'T2 = {T2}')
# -

# ## Pi function

temp1 = np.array([T1, (T1+T2)/2, T2])
temp2 = np.array([pi0, pi_max, pi0])
f = interp1d(temp1, temp2,  kind='quadratic',fill_value='extrapolate')
plt.plot(t[T1:T2+1], f(t[T1:T2+1]))


def pi_t(t, pi0):
    if t <= T1 or t>= T2:
        pi = pi0
    else:
        pi = f(t)
    
    return pi
pi_t = np.vectorize(pi_t)


# +
# tau = T-t ==> t = T - tau

def pi_tau(tau, pi0):
    t = T2 - tau
    if t <= T1 or t>= T2:
        pi = pi0
    else:
        pi = f(t)
    
    return pi
pi_tau = np.vectorize(pi_tau)
# -

(pi_t(T, pi0) < alpha*theta/(1-theta) - rho) & (pi_t(T,pi0)>= -rho)

plt.plot(t, pi_t(t,pi0))
plt.xlabel('t')
plt.ylabel('$\pi_t$')
plt.title('$\pi_t$')
plt.grid(True)

# ## Adot = 0 locus

# +
# Adot = 0 일 때, 각 pi 값에 따른 As의 값을 구하기

Ygrid = np.zeros(len(t))

for idx, val in enumerate(pi_t(t, pi0)):
#     print(idx)
    Y = Symbol('Y')
    Y_sol = solve( alpha*theta*(u_prime(Y)-1)/((1-theta)*u_prime(Y)+theta) - rho - val, Y)
    if len(Y_sol) == 1:
        Ygrid[idx] = Y_sol[0]
    elif len(Y_sol) == 2:
        Ygrid[idx] = Y_sol[1]
        
# (18), (19) 사용해서 y에서 a구하기
As_list = (1-theta)*u(Ygrid) + theta*Ygrid
# -

plt.plot(t, As_list, label = '$\dot{A}=0$')
plt.grid(True)
plt.xlabel('t')
plt.ylabel('A')
plt.title('$\dot{A}=0$')
plt.legend()

As_list[T2] # 끝값

# ## Solving ODE

F = lambda t,A: A * ( -rho - pi_tau(t, pi0) + alpha*theta*(  ( u_prime(y_A_forODE(A))-1 ) / ( (1-theta)*u_prime(y_A_forODE(A)) + theta )  )      ) 
sol = solve_ivp(fun = F, t_span = [0,T2], y0 = [As_list[T2]], method = 'RK45', t_eval = np.arange(0, T2, T2/100)) 
# 그래프를 더 부드럽게 그리고 싶으면 t_eval에 추가해주기 

# 거꾸로 푼 것이기 때문에 그래프도 거꾸로 읽어야함
plt.plot(sol.t, sol.y[0])
plt.xlabel('T-t')
plt.ylabel('$A_{T-t}$')
plt.grid(True)
plt.title('$A_{T-t}$')

# +
plt.plot(np.flip(T2-sol.t), np.flip(sol.y[0]), label = '$A_t$')

plt.plot(t, As_list, label = '$\dot{A}=0$', color ='red')
plt.grid(True)
plt.title('Time path of $A_t$')
plt.xlabel('t')
plt.ylabel('$A_t$')
plt.legend()

# -

# ## Discrete하게 풀기

# +

A0 = As_list[T2]

A_path = []
A_path.append(A0)

i = 0
A = A0

while i < T2:
    pi = pi_tau(i, pi0)
    Anew = find_next_A(A, pi)
    A_path.append(Anew)
    # update 
    A = Anew
    i += 1
# -

plt.plot(A_path, label = 'discrete')
plt.plot(sol.t, sol.y[0], label = 'continuous')
plt.grid(True)
plt.legend()

# ## 증가율 확인

# +
# 각 pi에 따른 Adot/A 그래프

y_A_vec = np.vectorize(y_A)
Adot_over_A_list = []
Amin = 0
Amax = 5
nA = 100
Agrid = np.linspace(Amin, Amax, nA)

for pi in pi_t(t, pi0):
    Adot_over_A = rho + pi - alpha*theta*(u_prime(y_A_vec(Agrid))-1)/((1-theta)*u_prime(y_A_vec(Agrid))+theta)
    Adot_over_A_list.append(Adot_over_A)
# -

T1

T2

# +
A_t = np.flip(sol.y[0])

# A0랑 A1 두개만 그려본 것
plt.plot(Agrid, Adot_over_A_list[5], label = 'pi5')
plt.plot(Agrid, Adot_over_A_list[8], label = 'pi8')
plt.plot(Agrid, Adot_over_A_list[20], label = 'pi20')
plt.plot(Agrid, Adot_over_A_list[39], label = 'pi39')

plt.grid(True)

plt.plot(A_t[5], 0, marker='o', label = 'A5')
plt.plot(A_t[8], 0, marker='o', label = 'A8')
plt.plot(A_t[20], 0, marker='o', label = 'A20')
plt.plot(A_t[39], 0, marker='o', label = 'A39')

plt.legend(loc=4)
plt.xlim((0,1))

# +
# 각 i기마다 A_i가 Adot_over_A(pi_i) 그래프를 따라서 A_(i+1)로 변하는지 확인 (증가율 확인)
# discrete하게 구한 값에 대하여

A_path_flip = np.flip(A_path) # 시간순으로 바꿔주기

for i in np.arange(0,T,1):
    dA = A_path_flip[i]*(rho + pi_t(i, pi0) - alpha*theta*(u_prime(y_A_vec(A_path_flip[i]))-1)/((1-theta)*u_prime(y_A_vec(A_path_flip[i]))+theta))
    A_next = A_path_flip[i] + dA
    print(A_next - A_path_flip[i+1])
    

# +
# 각 i기마다 A_i가 Adot_over_A(pi_i) 그래프를 따라서 A_(i+1)로 변하는지 확인 (증가율 확인)
# continuous하게 구한 값에 대하여

for i in np.arange(0,T,1):
    dA = A_t[i]*(rho + pi_t(i, pi0) - alpha*theta*(u_prime(y_A_vec(A_t[i]))-1)/((1-theta)*u_prime(y_A_vec(A_t[i]))+theta))
    A_next = A_t[i] + dA
    print(A_next - A_t[i+1])
# -






