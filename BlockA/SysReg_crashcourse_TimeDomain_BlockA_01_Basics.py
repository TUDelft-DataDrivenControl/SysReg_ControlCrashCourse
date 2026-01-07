# %%
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({ 'text.usetex':        False,              'mathtext.fontset':         'cm',
                      'font.size':          12.0,               'axes.labelsize':           'medium',
                      'xtick.labelsize':    'x-small',          'ytick.labelsize':          'x-small',
                      'axes.grid':          True,               'axes.formatter.limits':    [-3, 6],
                      'grid.alpha':         0.5,                'figure.figsize':           [11.0, 4],
                      'figure.constrained_layout.use': True,    'scatter.marker':           'x',
                      'animation.html':     'jshtml'})

from IPython.display import display, Markdown

import warnings
warnings.filterwarnings("ignore")

import scipy.signal as signal
import scipy.linalg as sclin
import numpy.random as rng
import numpy.linalg as lin
import control as cm
from helperFunctions import *

# %% [markdown]
# # Time domain
# ## Systems and differential equations
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*The basics of all things control*
# 
# *Systems are all and all is a system.* So you can decribe dynamic processses with physics, then you get equations with time derivatives. So for Linear-Time-Independent (LTI) systems, the differential equation with output $y(t)$ and input $u(t)$ is
# $$ \frac{d^n}{d^nt}y + a_1\frac{d^{n-1}}{d^{n-1}t}y + \cdots + a_n y = \frac{d^m}{d^mt}u + b_1\frac{d^{m-1}}{d^{m-1}t}u + \cdots + b_m u.$$
# Here, $n$ is called the order of the system.
# 
# We want to be able to solve Initial Value Problems (IVPs), because that's cool and it shows us what the system will do from some initial state. This solution is built in two steps, the homogenous solution and satisfying the initial conditions.
# 
# The homogenous solution is obtained by equation the left hand side to zero. For this, we'll steal a little ahead from the Laplace transform where
# $$ \frac{d^n}{d^nt} \overset{\mathcal{F}}{\rightarrow} s^n. $$
# Applying this and eliminating $y$ to obtain the homogenous solution yields
# $$\frac{d^n}{d^nt}y + a_1\frac{d^{n-1}}{d^{n-1}t}y + \cdots + a_n y = 0 $$
# $$\overset{\mathcal{F},\; \frac{1}{y}}{\rightarrow} s^n + a_1 s^{n-1} + \cdots + a_n = 0. $$
# This is a polynomial of order $n$ called **the characteristic polynomial** and we know then it has $n$ roots as well.
# 
# These roots, $\lambda_k$, actually form the solution to the homogenous problem, since a polynomial is the product of its roots,
# $$ \prod_{k=1}^n (s-\lambda_k).$$
# The solution to these types of ODEs are sums of exponentials of the form $c_k e^{\lambda_k t}$, where $c_k$ is determined through the initial condition. Therefore, the complete solution is
# $$ y(t) = \sum_{k= 1}^n c_k e^{\lambda_k t} .$$
# 
# If all $\mathfrak{R}(\lambda_k) < 0$ then $y(t)$ goes to 0 and the system is stable. The Routh-Hurwitz criterion gives the stability requirements up to the third degree. Stability is guaranteed for systems of order
# 1. if $a_1 > 0$.
# 2. if $a_1, a_2 > 0$.
# 3. if $a_1, a_2, a_3 >0$ and $ a_1a_2 > a_3$.
# 
# Real-valued $\lambda_k$ yield an exponential trajectory $e^{\lambda_k t}$. Complex-values come in conjugate pairs so they yield an exponential trajectory multiplied by a cosine. Because $\lambda_k, \lambda_{k+1} = \epsilon \pm j\omega$, the trajectory of this pair becomes $e^{(\epsilon \pm j\omega)t} = e^{\epsilon t} (e^{j\omega t} + e^{-j\omega t}) = e^{\epsilon t} (2\cos(\omega t))$. Play around with the code block here to get a feel for it!
# 

# %%
## Eigenvalues
sig1 = -0.8
sig2 = 3 + 15j
sig3 = -0.5 + 5j

###### Plotting #########
fig, ax = plt.subplots(1,3)
for sig, idx in zip([sig1, sig2, sig3], range(3)):
    t = np.linspace(0, 3/abs(np.real(sig)), num=300) # Adapt to convergence speed
    if np.iscomplex(sig): # Plot decomposition
        l1, = ax[idx].plot(t, np.exp(sig.real * t), 'r--', label=f"$e^{r"{"}{sig.real}t{r"}"}$") # Upper envelope
        ax[idx].plot(t, -np.exp(sig.real * t), 'r--') # Lower envelope
        l2, = ax[idx].plot(t, 2*np.cos(sig.imag * t), 'k', alpha=0.2, label=f"$2\cos({sig.imag} t)$") # Oscillation
        ax[idx].legend(handles=[l1, l2])
    ax[idx].plot(t, np.exp(sig * t), 'k') # Trajectory
    ax[idx].set(title=f"$\lambda = {sig}$", xlabel="$t$")

ax[0].set_ylabel("$y(t)$")
display(fig)


# %% [markdown]
# ## State space representation
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*A blessing from above*
# 
# Cool we know systems now, but it's a bit of an ugly system representation to be honest. Luckily we live in a universe where the state space representation exists and we can recast any system as a vector first order ODE and an output equation. This representation is of the form $$ \dot x = f(x, u)$$ $$ y = h(x, u),$$
# where $x$ is the state of the system, $u$ the input, and $y$ the output. The state dynamics are described with $f: \mathbb{R}^n\times\mathbb{R}^p\rightarrow\mathbb{R}^n$ and the output measurements with $h: \mathbb{R}^n\times\mathbb{R}^p\rightarrow\mathbb{R}^q$.
# 
# What black magic do we perform to get these first order ODEs? Well, suppose you have a second order ODE in $v$, $\ddot v = \dot v + v$, then this is equivalent to the first order ODE $$ \begin{bmatrix}\dot v\\\ddot v\end{bmatrix}  = \begin{bmatrix}0&1\\1&1\end{bmatrix}\begin{bmatrix}\dot v\\ v\end{bmatrix}.$$ 
# 
# Last but not least of the amazing aspects of the state space representation: there are many nice numerical integrators to simulate them. Think forward Euler or Runge-Kutta.

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture 2 (Lecture 1 was course logistics)</div>

# %% [markdown]
# ### Block diagrams
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Pixel perfect ways to visualise systems*
# 
# ![Slide about block diagram elements](figures/BlockDiagEls.png)
# 
# It's a nice sanity check that any fundamental block scheme of an $n$-th order system has $n$ integrators.
# 
# 
# ## Equilibrium points
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;*Forever and unchanging*
# 
# Often, systems will have equilibria, meaning there are states where system will remain over time. These state-input point pairs are denoted as $(x_e, u_e)$ or $(\bar x, \bar u)$. I'll use the latter since this is more traditional in control and we have to appease our control elders (JW). So equilibria remain unchanging, so the derivative of the state is zero. In mathematicians' language
# $$ \dot{\bar x} = f(\bar x, \bar u) = 0. $$
# Finding the equilibria states as a function of the equilibria inputs is as simple as solving this equation.
# 
# 
# 
# 

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture 3</div>
