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

# # Application 1-1_new inflation target

# ## Setting

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import random
from scipy.integrate import solve_ivp 
from sympy import *
from sympy.solvers import solve


# ### Functions

def u(y):
    u = ((y+b)**(1-eta) - b**(1-eta))/(1-eta)
    return u


def u_prime(y):
    u = 1/(y+b)**eta
    return u


# +
# Find steady-state value of A depending on $\pi$ by using (22)
# ODE 풀 때, initial value 설정할 때 필요함

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

# tau version (28-1)식을 discrete하게 구하는거 
def find_next_A(A): 
    A_next = A + A*(-rho -pi_0 + alpha*theta*(  ( u_prime(y_A(A))-1 ) / ( (1-theta)*u_prime(y_A(A)) + theta )))
    return A_next


# ### Parameters

# +
# parameters 
b = 0.01
eta = 0.5
rho = 0.04
theta = 0.5 # buyer's bargaining power
alpha = 0.5

pi_0 = 0
pi_1 = 0.1
# -

As_1 = find_steadystate_A(pi_1)

As_0 = find_steadystate_A(pi_0)

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

# ## Solving ODE (28-1)

# +
# Adot (= dA/dt) = F 의 형태로 정리

F = lambda t,A: A * (     -rho -pi_0 + alpha*theta*(  ( u_prime(y_A_forODE(A))-1 ) / ( (1-theta)*u_prime(y_A_forODE(A)) + theta )  )      )
# -

# ### T = 1
# - 시간이 거꾸로 되어있다는 점을 감안하고, 그래프를 오른쪽에서 왼쪽으로 읽어야함
# - T-t = 1, 즉 t=0일 때 보면 As_0에서 jump가 일어난 것을 알 수 있음
# - T-t = 0, 즉 t=T일 때 보면 As_1, new steady state으로 수렴한 것을 알 수 있음
#

# +
T = 1
sol = solve_ivp(fun = F, t_span = [0,T], y0 = [As_1], method = 'RK45', t_eval = np.arange(0, T, T/100)) # np.arange(0,T+1,1), np.arange(0, T, T/100)

# What is t_eval?
# Times at which to store the computed solution, must be sorted and lie within t_span. 
# If None (default), use points selected by the solver.
# t_eval은 솔루션을 저장하는 t의 포인트들을 찝어주는 것
# T가 고정일 때, t_eval을 몇개로 나누는지에 따라서 솔루션이 달라지지 않음
# -

# #### Evoluation of A

# +
plt.plot(sol.t, sol.y[0]) # sol.y가 A^T_t
plt.xlabel(f'$tau = {T}-t$')
plt.ylabel(f'$A^{T}_t$')
plt.grid(True)
plt.title('Reverse Evolution of A')
plt.plot(0.0, As_1, marker='*', label = 'As_1')
plt.axhline(y = As_0, color='gray', linestyle='--', label = 'As_0')
plt.legend(loc=0)

# 근데 proposition 4 보면, For all t < T, Adot < 0 인데... 이 부분 다시 확인 필요
# -

# #### Evolution of Y

# +
# (18), (19)를 사용해서 aggregate A가 주어지면 aggregate Y를 구할 수 있었음
Agrid = sol.y[0]

Ygrid = np.zeros(len(Agrid))
Ygrid[Agrid >= (1-theta)*u(y_star) + theta*y_star] = y_star # by (19)

for idx, val in enumerate(Agrid[Agrid < (1-theta)*u(y_star) + theta*y_star]): # by (18)
    Y = Symbol('Y')
    Y_sol = solve((1-theta)*u(Y) + theta*Y - val, Y)
    
    if len(Y_sol) == 1:
        Ygrid[idx] = Y_sol[0]
    elif len(Y_sol) == 2:
        Ygrid[idx] = Y_sol[1]
#     print(idx)


plt.plot(sol.t ,Ygrid)
plt.grid(True)
plt.xlabel(f'$tau = {T}-t$')
plt.ylabel(f'$y^{T}_t$')
plt.title('Reverse evolution of y')

# -

# #### Discrete

# +
# discrete하게 구해보기
# 얘도 backward로 푸는 것임

A0 = As_1

A_path = []
A_path.append(A0)

# 초기값 설정
i = 0
A = A0

while i < T:
    i += 1
    Anew = find_next_A(A)
    A_path.append(Anew)
    
    # update
    A = Anew 
    
plt.plot(A_path, label = 'discrete')
plt.plot(sol.t, sol.y[0], label = 'continuous')
plt.grid(True)
plt.legend()
plt.title('Reverse evoluation of A') # 오른쪽에서 왼쪽으로 읽어야함

# 완전히 똑같이 나오지는 않는데, 그게 혹시 tdot = 1로 똑같이 맞춰주지 않아서 그런건가 해서 continuous ODE에다가 t_eval을 
# tdot = 1 간격씩 솔루션을 저장하도록 지정해줘도 그래프가 똑같이 다르게 나옴 
# 지금 생각으로는, discrete은 딱딱 1에 맞춰서 솔루션을 풀지만, continuous는 그렇지 않기 때문인 것 같음
# 근데 discrete이랑 이렇게 비교하려면, T의 절대적인 숫자가 중요한게 아니라, continuous 0에서 T까지의 간격 갯수 = discrete에서의 T 값 이렇게 일치시켜줘야함
# 똑같이 T=1에서 계산을 하더라도, discrete에서는 그 사이에 연속적으로 어떻게 움직이는지는 알 수 없음. 
# 하지만 continuous로 풀면 T까지의 간격이 똑같이 1이어도 그 사이에 연속적인 움직임에 대해서 파악할 수 있음 
# 그럼 내 생각에.. continuous로 푼 값이 더 자세한 거 아닌가?
# -

A_path[-1] - sol.y[0][-1]

# #### $\dot{A}/A$ 그래프 그리기
# - Fig 5.
# - (28) 사용

A_t = np.flip(sol.y[0])

# A_0에서 jump가 얼마나 일어나는지 
As_0 - A_t[0]

abs(As_1-A_t[-1]) # T일 때 As_1으로 도달하는지, 차이가 0이면 도달한 것

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
Adot_over_A_1 = pi_1 + rho - alpha*theta*(u_prime(Ygrid)-1)/((1-theta)*u_prime(Ygrid)+theta)



# +
plt.plot(Agrid, Adot_over_A_0, label = '$\pi_0$')
plt.plot(Agrid, Adot_over_A_1, label = '$\pi_1$')

plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.title('Phase diagram: From start to end')
plt.plot(As_0, 0.0, marker='*', label = 'As_0')
plt.plot(A_t[0], 0, marker ='o', color ='red', label = 'start position (A0)')
plt.plot(A_t[-1],0,marker ='o',color='blue', label = 'End point')

# plt.plot(A_path[0], 0, marker = '^', color = 'yellow', label = 'discrete end')
# plt.plot(A_path[-1], 0, marker = '^', color = 'black', label = 'discrete start')


plt.legend(loc=4)


# +
# From initial jump to new steady-state

plt.plot(Agrid, Adot_over_A_0, label = '$\pi_0$')
plt.plot(Agrid, Adot_over_A_1, label = '$\pi_1$')

plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.title('Phase diagram: Moving')
plt.xlim((0.6,0.8))
plt.plot(As_0, 0.0, marker='*', label = 'As_0')
plt.plot(A_t[0], 0, marker ='o', label = 'end point')
plt.plot(A_t[-1], 0, marker ='o', color='b', label = 'start position')
plt.ylim((-0.4, 0.2))

for i in range(len(A_t)):
    plt.plot(A_t[i], 0, marker = 'o')
    
plt.legend(loc = 4)

# -

# ### T = 10
# - pi 가 한번 증가하는 시점인 T가 더 먼 시점일 때
# - Lemma 2 에서의 마지막 statement처럼 $\lim_{\tau \rightarrow \infty} A^R_\tau = A^s_0$
# - 첫번째 점프가 거의 일어나지 않음
#

T = 10
sol = solve_ivp(fun = F, t_span = [0,T], y0 = [As_1], method = 'LSODA', t_eval = np.arange(0,T+1,1)) # np.arange(0,T+1,1), np.arange(0, T, T/100))

plt.plot(sol.t, sol.y[0])
plt.xlabel(f'$tau = {T}-t$')
plt.ylabel(f'$A^{T}_t$')
plt.grid(True)
plt.title('Reverse Evolution of A')
plt.plot(0.0, As_1, marker='*', label = 'As_1')
plt.axhline(y = As_0, color='gray', linestyle='--', label = 'As_0')
plt.legend(loc=0)

# +
# discrete하게 구해보기
T = 10

A0 = As_1

A_path = []
A_path.append(A0)

# 초기값 설정
i = 0
A = A0

while i < T:
    i += 1
    Anew = find_next_A(A)
    A_path.append(Anew)
    
    # update
    A = Anew 
    
plt.plot(A_path, label = 'discrete')
plt.plot(sol.t, sol.y[0], label = 'continuous')
plt.grid(True)
plt.legend()

# -

A_path[-1] - sol.y[0][-1]

# +
plt.plot(Agrid, Adot_over_A_0, label = '$\pi_0$')
plt.plot(Agrid, Adot_over_A_1, label = '$\pi_1$')

plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.title('Phase diagram: From start to end')
plt.plot(As_0, 0.0, marker='*', label = 'As_0')
plt.plot(sol.y[0][-1], 0, marker ='o', color ='red', label = 'start position (A0)')
plt.plot(sol.y[0][0],0,marker ='o',color='blue', label = 'End point')

plt.plot(A_path[0], 0, marker = '^', color = 'yellow', label = 'discrete end')
plt.plot(A_path[-1], 0, marker = '^', color = 'black', label = 'discrete start')


plt.legend(loc=4)


# +

    
    
plt.plot(Agrid, Adot_over_A_0, label = '$\pi_0$')
plt.plot(Agrid, Adot_over_A_1, label = '$\pi_1$')

plt.grid(True)
plt.xlabel('A')
plt.ylabel('$\dot{A}/A$')
plt.title('Phase diagram: Moving')
plt.plot(As_0, 0.0, marker='*', label = 'As_0')

for i in range(len(sol.y[0])):
    plt.plot(sol.y[0][i], 0, marker = 'o', color = 'red')

for i in range(len(A_path)):
    plt.plot(A_path[i], 0, marker = '^', color = 'blue')
    
# plt.plot(sol.y[0][-1], 0, marker ='o', color ='red', label = 'start position (A0)')
# plt.plot(sol.y[0][0],0,marker ='o',color='blue', label = 'End point')

# plt.plot(A_path[0], 0, marker = '^', color = 'yellow', label = 'discrete end')
# plt.plot(A_path[-1], 0, marker = '^', color = 'black', label = 'discrete start')


plt.legend(loc=4)

# -

# #### Discrete

# +
# discrete하게 구해보기
# 얘도 backward로 푸는 것임

A0 = As_1

A_path = []
A_path.append(A0)

# 초기값 설정
i = 0
A = A0

while i < T:
    i += 1
    Anew = find_next_A(A)
    A_path.append(Anew)
    
    # update
    A = Anew 
    
plt.plot(A_path, label = 'discrete')
plt.plot(sol.t, sol.y[0], label = 'continuous')
plt.grid(True)
plt.legend()
plt.title('Reverse evoluation of A') # 오른쪽에서 왼쪽으로 읽어야함

# 완전히 똑같이 나오지는 않는데, 그게 혹시 tdot = 1로 똑같이 맞춰주지 않아서 그런건가 해서 continuous ODE에다가 t_eval을 
# tdot = 1 간격씩 솔루션을 저장하도록 지정해줘도 그래프가 똑같이 다르게 나옴 
# 지금 생각으로는, discrete은 딱딱 1에 맞춰서 솔루션을 풀지만, continuous는 그렇지 않기 때문인 것 같음
# 근데 discrete이랑 이렇게 비교하려면, T의 절대적인 숫자가 중요한게 아니라, continuous 0에서 T까지의 간격 갯수 = discrete에서의 T 값 이렇게 일치시켜줘야함
# 똑같이 T=1에서 계산을 하더라도, discrete에서는 그 사이에 연속적으로 어떻게 움직이는지는 알 수 없음. 
# 하지만 continuous로 풀면 T까지의 간격이 똑같이 1이어도 그 사이에 연속적인 움직임에 대해서 파악할 수 있음 
# 그럼 내 생각에.. continuous로 푼 값이 더 자세한 거 아닌가?
# -




