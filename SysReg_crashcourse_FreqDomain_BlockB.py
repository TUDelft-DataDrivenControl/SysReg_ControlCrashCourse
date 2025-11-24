
# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Transfer Functions"</div>

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Bode Plots"</div>

# %% [markdown]
# Put in the Nyquist explaination here (now at the end of the document)
# <div style="text-align:center;background-color:tomato;">End of lecture "Frequency Domain Analysis"</div>

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "PID control"</div>

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Frequency Domain Design I & II"</div>

# %% [markdown]
# ---

# %% [markdown]
# ## Introduction to the Nyquist Criterion
# 
# This exploration will be top-down, so we'll start with the definition and work our way down to the nitty gritty of why and how the criterion works.
# 
# ### Definition
# Wikipedia gives this definition of the criterion:
# 
# >Given a Nyquist contour $\Gamma_s$, let $P$ be the number of poles of $L(s)$ encircled by $\Gamma_s$, and $Z$ be the number of zeros of $1+L(s)$ encircled by $\Gamma_s$. Alternatively, and more importantly, if $Z$ is the number of poles of the closed loop system in the right half plane, and $P$ is the number of poles of the open-loop transfer function $L(s)$ in the right half plane, the resultant contour in the $L(s)$-plane, $\Gamma_{L(s)}$, shall encircle (clockwise) the point $(-1+j0)$ $N$ times such that $N = Z - P$.
# 
# That's pretty overwhelming! Let's take it apart:
# - Nyquist contour $\Gamma_s$: this is defined as the clockwise contour encompassing the Right Half of the imaginary Plane (RHP). It is also the collection of all $s$ we'll consider.
# - $L(s)$: the loop transfer function; the product of the plant/process and controller transfer functions.
# - $P$: the number of RHP poles of $L(s)$.
# - $Z$: the number of RHP zeros of $1+L(s)$ **and** the number of RHP poles of the closed loop system.
# - $N$: the number of clockwise rotations around $-1+j0$ of $L(\Gamma_s)$, and $N=Z-P$.
# 
# ### Stability
# So $Z$ is supposedly the number of unstable poles of the closed loop system. Then if $Z=0$, there are no RHP poles destabilizing the closed loop, so the closed loop is stable! Now we also got that $N=Z-P\rightarrow Z=N+P$, so we can just count the clockwise encirclements in the Nyquist plot $L(\Gamma_s)$ and add it to the number of RHP poles of $L(s)$ and if that's zero the closed loop is stable! Wait but something added to something should be zero? Good question, $N$ is positive for clockwise encirclements, but negative for counterclockwise encirclements. So, alternatively, we can say the closed loop is stable if (and only if) the **net** number of counterclockwise encirclements of $-1+j0$ is equal to the number of RHP poles of $L(s)$. This is our main use for the Nyquist criterion, because this is how we assess stability.
# 
# ### Where do they get these transfer functions from?
# Another excellent question! This is where it gets a tad more difficult, but we'll soon look at some pictures too. First, we have a quick look at the closed loop transfer function again. Our block diagram with reference $r$, output error $e$, actuation $u$, output $y$, controller $C$, and process $G$ is 
# 
# ![](CLsys.svg)
# 
# Then in the frequency domain the transfer from the reference to the output is $\frac{Y(s)}{R(s)}$. We get this by following the chain as 
# $$Y(s) = G(s)U(s) = G(s)C(s)E(s) = L(s)E(s) = L(s)(R(s)-Y(s))$$
# $$\rightarrow (1+L(s))Y(s) = L(s)R(s) \rightarrow \frac{Y(s)}{R(s)} = \frac{L(s)}{1+L(s)}.$$
# 
# So we can conclude that zeros of $1+L(s)$ are poles of our closed loop system. We saw $1+L(s)$ before! *And they were talking about its zeros!!* In the Nyquist criterion, $Z$ was defined as "the number of zeros of $1+L(s)$ encircled by $\Gamma_s$" **and** "the number of poles of the closed loop system in the right half plane." We've now discovered why these are linked!
# 
# ### Why encirclements then?
# This is where we start letting go of formal mathematics and start going more by visuals. If you're cool however, you can find out more about the formalities in [Cauchy's argument principle](https://en.wikipedia.org/wiki/Argument_principle). It'll take a couple of steps to arrive at the source of the encirclements, but no worries. We'll go through it step by step. 
# 
# ### First the contour
# We'll not go for the full Nyquist contour $\Gamma_s$ immediately, but first take a smaller contour $\Gamma$. Lets define that!

# %%
import scipy.interpolate as interp

def Gamma(Q, R):
    # Q in [0, 1], contour coordinate
    # R >0, radius of semicircle contour
    contourLength = 2*R + np.pi*R**2 /2 # diameter + halfcircle
    corner = R / contourLength # the contour coordinate of the corner
    cntr = np.empty_like(Q).astype(complex)
    
    cntr[Q <= corner] = Q[Q <= corner]/corner* R*1j # we're on the positive imaginary axis
    cntr[Q >= 1-corner] = (1-Q[Q >= 1-corner])/corner* R*-1j # we're on the negative imaginary axis
    angMap = interp.interp1d([corner, 1-corner], [np.pi/2, -np.pi/2])
    semcirIdx = (Q>corner) * (Q<1.0-corner) # we're on the semi circle
    cntr[semcirIdx] = R*np.exp(angMap(Q[semcirIdx])*1j)

    return cntr

def drawContour(ax, cntr):## Plot contour
    ax.plot(np.real(cntr), np.imag(cntr), '--', color=[0.1,0.7,0.8])
    ax.set(aspect='equal', ylabel="Im", xlabel="Re")

    ## Plot clockwise arrows
    arrowIdx = np.round(np.linspace(start=0, stop=len(Q), num=6, endpoint=False)).astype(int)

    for idx in range(len(arrowIdx)):
        ax.annotate("", xytext=(np.real(cntr[arrowIdx[idx]]), np.imag(cntr[arrowIdx[idx]])), 
                    xy=(np.real(cntr[arrowIdx[idx]+1]), np.imag(cntr[arrowIdx[idx]+1])),
                    arrowprops=dict(arrowstyle="->"))
    


# %%
## Create contour
Q = np.linspace(start=0, stop=1, num=150, endpoint=True)
R = 2
cntr = Gamma(Q, R)

fig1, ax = plt.subplots()
drawContour(ax, cntr)

plt.show()

# %% [markdown]
# That's a nice semicircle of finite radius! In the end we want to 'walk along the contour' to get its mapping. So we have our complex variable $s$ for our transfer functions and this we'll walk along the contour. That looks like this:

# %%
import matplotlib.animation as animation

fig1, ax = plt.subplots()
drawContour(ax, cntr)
sc = ax.scatter([np.real(cntr[0])], [np.imag(cntr[0])], marker='x', color='red')
ax.legend([r'$\Gamma$', r'$s$'])

def animFun(t): 
    sc.set_offsets([np.real(cntr[t]), np.imag(cntr[t])])
    return sc, 
anim = animation.FuncAnimation(fig1, func=animFun, frames=len(Q), interval=30, blit=True)

from IPython.display import HTML
# plt.close(fig1)
# HTML(anim.to_jshtml())

# %% [markdown]
# ### Shuffling $L(s)$
# Next, we will demonstrate the origin of the encirclements. We need to rewrite $L(s)$ a little first though. For a Linear Time Invariant (LTI) system the loop function can be written as a fraction of two polynomials with the respective roots being the zeros and poles of the loop function. Many words, this is it in maths
# $$ L(s) = \frac{N_L(s)}{D_L(s)} $$ 
# and 
# $$ 1+L(s) = \frac{N_L(s) + D_L(s)}{D_L(s)}.$$
# From this we can see that $L(s)$ and $1+L(s)$ share the same poles, which are the roots of $D_L(s)$.
# 
# Suppose we call $1+L(s) = F(s)$ for now. We are interested in the zeros of $F(s)$, since these are our closed loop poles. Also we know that the sum of two polynomials is another polynomial, so we can write
# $$ F(s) = \frac{N_L(s) + D_L(s)}{D_L(s)} = \frac{N_F(s)}{D_F(s)}.$$ 
# The roots of $N_F(s)$ are therefore zeros of $F(s)$.
# 
# ### Polynomials
# We'll quickly jump into polynomials for a second. Polynomials are largely defined by their roots. For example, this is what we do when we factorize a parabola formula to find its roots. Similar to that, we can write a polynomial $N_F(s)$ as the product of its $Z$ number of roots, $z_k$, as
# $$ N_F(s) = (s - z_1)(s-z_2)...(s-z_Z) = \prod_{k=1}^Z (s - z_k).$$
# Then we can also define $N_F^\dagger (s) = \prod_{k=2}^Z (s - z_k)$, such that 
# $$N_F(s) = (s-z_1)N_F^\dagger (s).$$
# 
# ### Inside or outside
# Now we're going to make two categories: zeros inside $\Gamma$ and zeros outside $\Gamma$. Let's say $z_1$ lies outside $\Gamma$ and look at the behaviour of *the phase of* $(s-z_1)$ when we walk $s$ along $\Gamma$.

# %%
from matplotlib.patches import Arc

def anim_init(z1, ax):
    for i in range(2):
        ax[i].scatter(np.real(z1), np.imag(z1), marker='o', facecolors='none', edgecolors='r', lw=1)
        ax[i].annotate("$z_1$", xy=(np.real(z1), np.imag(z1)),
                    xytext=(.5,.3), textcoords='offset fontsize', color='red')

    ## Animation time! Yeah don't focus too much on this it's fine
    r = cntr - z1
    rm = np.max(np.abs(r))
    ax[1].set(aspect='equal', ylabel="Im", xlabel="Re", xlim=[-rm+np.real(z1), rm+np.real(z1)], ylim=[-rm+np.imag(z1),rm+np.imag(z1)])

    arr = [None,None]
    for i in range(2):
        arr[i] = ax[i].annotate("", xy=(np.real(cntr[0]), np.imag(cntr[0])), 
                    xytext=(np.real(z1), np.imag(z1)),
                    arrowprops=dict(arrowstyle="-|>", color='red'))

    phl1 = plt.plot([np.real(z1),rm/2+np.real(z1)], [np.imag(z1),np.imag(z1)], 'k', lw=1.2, alpha=0.3)
    phl2 = plt.plot([np.real(z1),np.real(z1)+np.real(rm/2*np.exp(1j*np.angle(r[0])))], 
                    [np.imag(z1),np.imag(z1)+np.imag(rm/2*np.exp(1j*np.angle(r[0])))], 'k', lw=1.2, alpha=0.3)
    phArc = Arc((np.real(z1),np.imag(z1)), width=rm/2.3, height=rm/2.3, theta2=np.angle(r[0], deg=True))
    ax[1].add_patch(phArc)

    angr = np.angle(r)
    angr[angr<0] += 2*np.pi
    phlims = np.array([rm/3*np.exp(np.min(angr)*1j), 0j, rm/3*np.exp(np.max(angr)*1j)]) + z1
    phl3 = plt.plot(np.real(phlims), np.imag(phlims), 'b', lw=1.2, alpha=0.3)
    return arr, phArc, phl2, r, rm

def animate(t):
    for i in range(2):
        arr[i].xy = (np.real(cntr[t]), np.imag(cntr[t]))
    phArc.theta2 = np.angle(r[t], deg=True)
    phl2[0].set_data([np.real(z1),np.real(z1)+np.real(rm/2*np.exp(1j*np.angle(r[t])))], 
                    [np.imag(z1),np.imag(z1)+np.imag(rm/2*np.exp(1j*np.angle(r[t])))])
    return arr[0], arr[1], phArc, phl2[0]


# %%

fig2, ax = plt.subplots(1, 2)
drawContour(ax[0], cntr)

## define the zero
z1 = 1.3*R*np.exp(-2/7*np.pi*1j) # You can change me if you'd like

arr, phArc, phl2, r, rm = anim_init(z1, ax)
anim = animation.FuncAnimation(fig2, func=animate, frames=len(Q), interval=50, blit=True)


# plt.close(fig2)
# HTML(anim.to_jshtml())

# %% [markdown]
# Okay, that's looking cool! The angle doesn't seem to be changing all that much, but it's funny to look at I guess. Let's move that zero into $\Gamma$ now and see what happens.

# %%
fig3, ax = plt.subplots(1, 2)
drawContour(ax[0], cntr)

## define the zero
z1 = 0.8*R*np.exp(-2/7*np.pi*1j) # You can change me if you'd like

arr, phArc, phl2, r, rm = anim_init(z1, ax)
anim = animation.FuncAnimation(fig3, func=animate, frames=len(Q), interval=50, blit=True)

# plt.close(fig3)
# HTML(anim.to_jshtml())

# %% [markdown]
# *Wait...* was that a circle? Actually, that makes sense right? $\Gamma$ goes around $z_1$, so then it has to make a full circle. Woah okay, so $(s-z_1)$ makes a full rotation if (and yes, only if) $z_1$ is on the inside of $\Gamma$, otherwise it just oscilates.
# 
# Cool, but this is only for one zero though? What about the others? Well, remember the polynomial
# $$ N_F(s) = \prod_{k=1}^Z (s - z_k)?$$
# It says here that $N_F(s)$ is the product of a bunch of complex numbers. We know what that means, adding the phases! So any zero inside $\Gamma$ adds a full clockwise encirclement to $N_F(\Gamma)$ (and also $F(\Gamma)$) around **the origin**. What do the outside zeros add though? There are two options: if the zero is real, it splits $\Gamma$ exactly in half and doesn't add any phase overall, if the zero is complex it does add some phase of course. However, complex zeros come in conjugate pairs, so these phase additions cancel out because they are mirrored over the real axis.
# 
# How do poles behave here actually? Very similar really, but since the poles are in the denominator of $F(s)$, they subtract phase rather than add. So poles inside $\Gamma$ cause counterclockwise encirclements of the origin.
# 
# ### I forgot what we were trying to figure out
# Yes, me too. Let's run it backwards. We know that zeros inside the contour add a full clockwise rotation and poles add a counterclockwise rotation. If we expand our finite $\Gamma$ to the infinite Nyquist contour $\Gamma_s$, this will encompass all right half plane (RHP) zeros and poles. So the number of clockwise encirclements of the origin of $F(\Gamma_s)$ equals the number of RHP zeros minus the RHP poles. That sounds very familiar! It's similar to the $N=Z-P$ equation, except that is shifted. Realise that
# $$F(\Gamma_s) = 1 + L(\Gamma_s),$$
# so encirclement of $0+j0$ by $F(\Gamma_s)$ means encirclement of $-1 + j0$ by $L(\Gamma_s)$.
# 
# And that's nearly it! Looking back at the practically applicable Nyquist criterion:
# > The closed loop is stable if (and only if) the net number of counterclockwise encirclements of $-1+j0$ is equal to the number of RHP poles of $L(s)$.
# 
# We're missing one last step. Since $L(s)$ and $F(s)$ share the same poles, and the number of counterclockwise encirclements of $-1+j0$ is equal to the number of these poles in the RHP, the number of CCW encirclements and the number of RHP poles of $L(s)$ from the criterion are only equal if there are no RHP zeros of $F(s)$ adding clockwise encirclements. Then, since there are no RHP zeros of $F(s)$, the closed loop is stable!
# 
# And that is why Nyquist works in very rough strokes.
# 
# ### What if I'm freaky and use positive feedback?
# Then the definition of $F(\Gamma_s) = 1 - L(\Gamma_s)$, so then $N$ is the number of encirclements around $1 + j0$. But why would you do that?
# 
# 

# %% [markdown]
# ## Nyquist vs. Bode, epic rap battles of history
# 
# We've seen both Bode plots and the Nyquist plot now. However, just a recap: Bode plots have $s=j\omega$ on the horizontal axes and on the two vertical axes is the magnitude and phase of the transfer function. We've also been looking at magnitude and phase with Nyquist so we must be able to relate the two! Lets look at a dummy transfer function $G(s)$.
# 
# ---
# 
# I just make some plots down here, guiding text still necessary!!!

# %%
# cm.reset_rcParams()

s = cm.tf('s')
if True: # Toggle between manual and random
    poles = [0j, -0.05+0.45j, -0.05-0.45j]
    zeros = [-0.6]
    G = 1
    for z in zeros: G *= (s-z)
    for p in poles: G *= 1/(s-p)
else:
    G = cm.rss(states=6)


# G *= 1/(s-0.3)

poles = cm.poles(G)
zeros = cm.zeros(G)
print("G(s) = ")
G


# %%
fig, ax = plt.subplots()
plt.autoscale(True)
ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='red', facecolors='none')
ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red')
ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
ax.fill_betweenx(ax.get_ylim(), [0], ax.get_xlim()[1], facecolor=[0, 1, 1], alpha=.3, hatch='/')
ax.scatter(np.real(zeros), np.imag(zeros), marker='o', color='red', facecolors='none')
ax.scatter(np.real(poles), np.imag(poles), marker='x', color='red')


# %%

Nyq = cm.nyquist_response(G)
# Nyq.plot()
r = np.logspace(-3, 2, 1000)*1j# Nyq.contour

G*=1
L = G(r)
uncircle = np.exp(np.linspace(0, np.pi*2)*1j)

fig, ax = plt.subplots(1,2)
for i in range(2):
    ax[i].plot(np.real(L), np.imag(L), color=[0, .4, .8])
    ax[i].plot(np.real(L), -np.imag(L), '--', color=[0, .8, .4])
    ax[i].scatter([-1], [0], marker='x', color='red')
    ax[i].scatter(np.real(G(0j)), np.imag(G(0j)), marker='o', color=[0, .8, .4])
    ax[i].plot(np.real(uncircle), np.imag(uncircle), 'r--', lw=0.5)

    # arrowIdx = np.round(np.linspace(start=0, stop=len(r), num=6, endpoint=False)).astype(int)

    # for idx in range(len(arrowIdx)):
    #     ax[idx].annotate("", xytext=(np.real(L[arrowIdx[idx]]), np.imag(L[arrowIdx[idx]])), 
    #                 xy=(np.real(L[arrowIdx[idx]+1]), np.imag(L[arrowIdx[idx]+1])),
    #                 arrowprops=dict(arrowstyle="->"))

ax[1].set(aspect='equal')
ax[1].set(xlim=(-4, 1), ylim=(-2,2))
Nyq.count

# %%
sysFB = cm.feedback(G)
T, yout = cm.step_response(G)
fig, ax = plt.subplots()
ax.plot(T, yout)


