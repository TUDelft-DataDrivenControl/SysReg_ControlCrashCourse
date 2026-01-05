# %% [markdown]
# <div style="text-align: right">Sachin Umans</div>
# <div style="text-align: right">November 2025</div>

# %% [markdown]
# # Crash course for the control uninitiated for Systeem & Regeltechniek

# %%
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({ 'mathtext.fontset':         'cm',
                      'font.size':          12.0,               'axes.labelsize':           'medium',
                      'xtick.labelsize':    'x-small',          'ytick.labelsize':          'x-small',
                      'axes.grid':          True,               'axes.formatter.limits':    [-3, 6],
                      'grid.alpha':         0.5,                'figure.figsize':           [11.0, 4],
                      'figure.constrained_layout.use': True,    'scatter.marker':           'x',
                      'animation.html':     'jshtml'
                    })

from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from IPython.display import display, Markdown

import warnings
warnings.filterwarnings("ignore")

import control as cm
from helperFunctions import *

# %% [markdown]
# # Frequency Domain
# Okay now stuff actually gets a little tricky so I'll have to be a bit more serious sadly.
# ## The Laplace (and Fourier) Transform is a godsent
# ### Transform of arbitrary signals
# Suppose you have a time-varying signal $u(t)$, then there exists its (bilateral) Laplace or Fourier transform $U(s)$, where $s = \sigma + i\omega$ for the Laplace transform and $s = i\omega$ for 
# the Fourier transform.
# 
# These transforms are defined as 
# 
# $$\mathcal{L}\{u(t)\}(s) = U(s) = \int_{-\infty}^\infty u(t) e^{-st} \text{d}t,\; s \in \mathbb{C}$$
# and
# $$\mathcal{F}\{u(t)\}(\omega) = U(\omega) = \int_{-\infty}^\infty u(t) e^{-i\omega t} \text{d}t,\; \omega \in \mathbb{R}.$$
# And now you know why it's called the frequency domain: $\mathfrak{Re}(e^{-st})$ is a (dampened) oscillation!
# To be completely honest, this is *way* too big of a subject for me to explain. However, I don't have to because of legends like 3Blue1Brown, go watch these if you want to know more:
# - [watch this video on what Fourier transforms are](https://www.youtube.com/watch?v=r6sGWTCMz2k)
# - [and this one on Laplace transforms.](https://www.youtube.com/watch?v=FE-hM1kRK4Y)
# - [Just watch his playlist on differential equations really, the man's a treasure.](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6)
# 
# But if you don't care about all that and just want to apply it, you need these key fact:
# - The Fourier transform is a restricted Laplace transform, but most things work for both so I'll just continue with the Laplace transform for now,
# - These transforms basically say that **any** time signal is a sum of (dampened) sinusoids,
# - Both transforms are linear.
# 
# Furthermore, some interesting mathematical fact:
# Starting with the definition again
# $$\mathcal{L}\{u(t)\}(s) = U(s) = \int_{-\infty}^\infty u(t) e^{-st} \text{d}t,\; s \in \mathbb{C}$$
# and integrating by part [(trust me bro)](https://en.wikipedia.org/wiki/Integration_by_parts)
# $$ \mathcal{L}\{u(t)\}(s) = \left[\frac{u(t) e^{-st}}{-s} \right]_{-\infty}^\infty + \frac{1}{s} \mathcal{L}\left\{\frac{\text{d} u(t)}{\text{d}t}\right\}(s) $$
# $$ \qquad\qquad = \left[-\frac{u(-\infty) e^{s\infty}}{-s} \right] + \frac{1}{s} \mathcal{L}\left\{\frac{\text{d} u(t)}{\text{d}t}\right\}(s) $$
# Now **assuming** $u(-\infty) = 0$, which really is not that bad of an assumption,
# $$ \mathcal{L}\{u(t)\}(s) = \frac{1}{s} \mathcal{L}\left\{\frac{\text{d} u(t)}{\text{d}t}\right\}(s).$$
# ... Amazing right! No but really this shows two major properties of the Laplace and Fourier transforms:
# $$ \mathcal{L}\left\{\frac{\text{d} u(t)}{\text{d}t}\right\}(s) = s \mathcal{L}\{u(t)\}(s) \Rightarrow \frac{\text{d}}{\text{d}t} U(s) = s U(s)\text{, and}$$
# $$ \mathcal{L}\left\{\int u(t)\text{d}t\right\}(s) = \frac{1}{s} \mathcal{L}\{u(t)\}(s) \Rightarrow \int U(s) \text{d}t = \frac{1}{s} U(s).$$

# ### Transfer functions
# Now *why* is that so amazing? Lets go back to the basics: the state space in time domain.
# $$ \dot x = Ax + Bu, $$
# $$ y = Cx + Du.$$
# Now taking that sweet Laplace transform:
# $$ s X(s) = AX(s) + BU(s), $$
# $$ Y(s) = CX(s) + DU(s).$$
# This allows us to look at something interesting: the ratio between the output and the input, called the *transfer function* $\frac{Y(s)}{U(s)} = G_{yu}(s)$. Pretty straightforward to derive:
# $$\rightarrow (sI -A) X(s) = BU(s) \rightarrow X(s) = (sI -A)^{-1} BU(s) $$
# $$ Y(s) = CX(s) + DU(s) \rightarrow Y(s) = C(sI -A)^{-1} BU(s) + DU(s) \rightarrow Y(s) = (C(sI -A)^{-1} B + D)U(s)$$
# $$ \Rightarrow \frac{Y(s)}{U(s)} = G_{yu}(s) = C(sI -A)^{-1} B + D$$
# This, however, is only for the steady state, if you also want the transient, it's in the slides.
# 
# For ODEs it's very similar, for example a 2nd order system (mass-spring-dampner systems are the textbook example):
# $$ m\ddot x + b\dot x + k x = \dot u - q u \overset{\mathcal{L}}{\rightarrow} (ms^2 + bs + k)X(s) = (s - q) U(s).$$
# $$ \rightarrow \frac{X(s)}{U(s)} = G_{xu}(s) = \frac{s - q}{ms^2 + bs + k}.$$
# 
# ### Gains, poles and zeros
# So a system can be fully defined by these three magical properties, which are:
# - The gain / steady state gain / DC gain: defined as the value of $G_{yu}(0)$,
# - The poles: the set of $s$-values for which $G_{yu}(s)$ is not defined, i.e. when $(sI -A)$ is not invertible or a division by 0 happens in the transfer function,
# - The zeros: the set of $s$-values for which $G_{yu}(s) = 0$. (zeros are underrated)

# ### Transfer functions for controls
# Lets look at a very general control scheme:
# 
# ![General feedback loop](figures/CLsys_general.svg)
# 
# with the block and signals:
# - $F$ : feedforward controller
# - $C$ : feedback controller
# - $P$ : the plant
# - $r$ : reference signal, global input
# - $e$ : tracking error
# - $u$ : control signal
# - $d$ : input disturbance
# - $v$ : disturbed input
# - $\eta$ : plant output
# - $n$ : output disturbance
# - $y$ : disturbed output, global output
# 
# I'm sure you're now able to do this, but I'll do it for you, the algebraic expressions for the error and output are
# $$ e = \frac{F}{1 + PC} r + \frac{-1}{1 + PC} n + \frac{-P}{1 + PC} d = G_{er}r + G_{en}n + G_{ed}d \text{, and}$$
# $$ y = \frac{PCF}{1 + PC} r + \frac{1}{1+PC} n + \frac{P}{1 + PC} d  = G_{yr}r + G_{yn}n + G_{yd}d.$$
# You might recognise the fractions as transfer functions and this reveals the robust control problem they might tackle in the master course: how to reject noise, but follow the reference?
# 
# <div style="text-align:center;background-color:tomato;">End of lecture "Transfer Functions"</div>

# %% [markdown]
# ## Visualising complex numbers: Bode plots
# Okay, so we can describe systems in the frequency domain with these funky transfer functions, so what? Well, we still have some way to go before we can start doing 
# useful stuff... I told you this was the part where it got messy.
# 
# Just to be clear, we have a transfer function $G(s)$ that maps the complex plane to the complex plane, i.e.
# $$ G: \mathbb{C} \rightarrow \mathbb{C}.$$
# Then we have the steady state output being $Y(s) = G_{yu}(s)U(s)$, so for a given $s$, $Y(s)$ is the product of two complex numbers $G_{yu}(s)$ and $U(s)$. Remember for complex numbers: multiplication means 
# multiplying the modulus/magnitude and adding the argument/angle/phase, i.e. $ae^{ci}\;be^{di} = abe^{(c+d)i}$. Likewise, division is division and subtraction, respectively, i.e. 
# $\frac{ae^{ci}}{be^{di}} = \frac{a}{b}e^{(c-d)i}$. These 
# two rules can be used to combine simple transfer function 'blocks' into more complicated functions. In any case, any input $U(s)$ will be amplified with $|G_{yu}(s)|$ and shifted with $\angle G_{yu}(s)$ to 
# the output $Y(s)$, because they're multiplied.
# 
# Now we want to visualise the behaviour of $G$, since that gives us a ton of information on that steady state input-output behaviour of the system. This can be done with *Bode* plots.
#  **Important: Bode plots use the Fourier transform, so $s=i\omega$ !**
# 
# So Bode follows the following rationale: $G$ maps the imaginary axis (pure sinusoids in time domain) to some other curve in the complex plane, not on the imaginary axis per
#  se. Any complex number can be defined through their modulus/magnitude/gain and argument/angle/phase, so plotting these against the oscillation frequency, $\omega$, defines the curve in the complex plane. 
# The gain is usually plotted on a log-log scale and the phase on a lin-log scale. Lets have a look at how that looks like for the simplest functions, $G = s^n$.

# %%
N = range(-2,3)
G1 = [lambda s, n=n : s**n for n in N]
OM = np.logspace(-2, 2)
S = OM*1j
G1_eval = [G(S) for G in G1]

### Plotting ###
fig = plt.figure(num="Bode plot intro")
gs = GridSpec(2,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[:, 0], gs[0, 1], gs[1, 1]]]

[ax[0].plot(G_eval.real, G_eval.imag, 'x' if n >= 0 else 'o', c='none', mec=f"C{idx}") 
        for G_eval, n, idx in zip(G1_eval, N, range(len(N)))]
hnd = [ax[1].loglog(OM, np.abs(G_eval), label=f"$s^{"{"}{n}{"}"}$")[0] 
       for G_eval, n in zip(G1_eval, N)]
[ax[2].semilogx(OM, np.angle(G_eval, deg=True)) for G_eval in G1_eval]
ax[0].set(title="Imaginary plane", 
          xlabel="$\mathfrak{Re}\{G(s)\}$", ylabel="$\mathfrak{Im}\{G(s)\}$", 
          xscale='symlog', yscale='symlog')
ax[1].set(title="Bode plot", ylabel = "$|G(s)|$")
ax[2].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[2].yaxis.set_major_locator(MultipleLocator(90))
ax[0].legend(handles=hnd)
display(fig)

# %% [markdown]
# What you should note in the plot above:
# - In the left plot the magnitude changes, but the phase is constant and how this translates to the Bode plot,
# - $s^0=1$ is constant and how that translates to the magnitude plot,
# - that in a log-log scale $s^n$, the magnitude is a line with slope $n$.
# 
# Now this plot was a bit boring, so lets make it more interesting! We'll plot the few simple transfer function 'blocks' as I mentioned earlier:
# - a constant gain, $k_1$,
# - a real pole, $p_1$,
# - a real zero, $z_1$,
# - a complex pole pair, with natural frequency $\omega_p$ and damping ration $\zeta_p$,
# - a complex zero pair, with natural frequency $\omega_z$ and damping ration $\zeta_z$,
# - a time-delay, $e^{-\tau_1 s}$, that delays with $\tau_1$ time

# %% 
k1, p1, z1 = 1e-2, 0.3, 7. #CHANGEME
om_p, zeta_p, om_z, zeta_z = .1, .3, 4., 2. #CHANGEME
tau1 = 3e-3 #CHANGEME

OM = np.logspace(-2, 3, 400)
S = OM*1j

s = cm.tf('s')
G2 = [lambda s : k1 * np.ones_like(s), # Constant gain
      lambda s : 1 / (s + p1), # Real pole
      lambda s : (s + z1), # Real zero
      lambda s : 1 / (s**2 + 2* zeta_p * om_p * s + om_p**2), # Comlex pole pair
      lambda s : (s**2 + 2* zeta_z * om_z * s + om_z**2), # Complex zero pair
      lambda s : np.exp(-tau1 * s)] # Time delay
G2_eval = [G(S) for G in G2]

### Plotting ###
fig = plt.figure(num="Bode plot blocks")
gs = GridSpec(2,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[:, 0], gs[0, 1], gs[1, 1]]]

for G_eval in G2_eval:
    ax[0].plot(G_eval.real, G_eval.imag)
    ax[1].loglog(OM, np.abs(G_eval))[0]
    ax[2].semilogx(OM, np.angle(G_eval, deg=True))

ax[0].legend([rf"${k1:.2f}$",
              rf"$\frac{"{"}1{"}"}{"{"}s + {p1:.2f}{"}"}$",
              rf"$s + {z1:.2f}$",
              rf"$\frac{"{"}1{"}"}{"{"}s^2 + 2 \cdot {zeta_p:.2f} \cdot {om_p:.2f}s + {om_p:.2f}^2{"}"}$",
              rf"$s^2 + 2 \cdot {zeta_z:.2f} \cdot {om_z:.2f}s + {om_z:.2f}^2$",
              rf"$e^{"{"}-{tau1:.3f}a{"}"}$",
              ], fontsize='small')

for G_eval, idx in zip(G2_eval, range(len(G2_eval))):
    ax[0].plot(G_eval[0].real, G_eval[0].imag, 'x', c=f"C{idx}")
    ax[1].loglog(OM[0], np.abs(G_eval[0]), 'x', c=f"C{idx}")
    ax[2].semilogx(OM[0], np.angle(G_eval[0], deg=True), 'x', c=f"C{idx}")

ax[0].set(title="Imaginary plane", 
          xlabel="$\mathfrak{Re}\{G(s)\}$", ylabel="$\mathfrak{Im}\{G(s)\}$", 
          xscale='symlog', yscale='symlog')
ax[1].set(title="Bode plot", ylabel = "$|G(s)|$")
ax[2].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[2].yaxis.set_major_locator(MultipleLocator(90))

display(fig)

# %% [markdown]
# That plot is a lot to take in, I do admit. However, the take-aways are:
# - poles and zeros are each other inverses,
# - the pole/zero blocks here all show three regions, when the constant term dominates, a transition region, and when the varying term part dominates,
# - a pole at a certain frequency initiates a slope down with steepness 1 and gains 90 degrees phase,
# - a zero at a certain frequency initiates a slope up with steepness 1 and loses 90 degrees phase,
# - therefore, any pole is a lowpass filter and any zero is a highpass filter,,
# - more therefore, connecting back to the Laplace transform: integrating ($\frac{1}{s}$) is a pole at $s=0$ and a lowpass filter, and differentiating ($s$) is a zero at $s=0$ and a highpass filter,
# - also, complex pairs double the steepness and phase change,
# - the behaviour of a complex pair in its transition region around the natural frequency depends on the damping ratio, we'll investigate that more now.
# 
# In this case we'll look at zero pairs, but you just learned that pole pairs just flip the zero pair bode plot.

# %%
OM = np.logspace(-2, 2, 400)
S = OM*1j

om1 = 1. 

fig = plt.figure(num="Damping ratio TF")
gs = GridSpec(2,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[:, 0], gs[0, 1], gs[1, 1]]]

hnd = []
for zeta in np.geomspace(0.05, 10., num=4):
    G_eval = (S**2 + 2* zeta * om1 * S + om1**2)
    
    ax[0].plot(G_eval.real, G_eval.imag)
    hnd.append(
        ax[1].loglog(OM, np.abs(G_eval), label = rf"$\zeta$ = {zeta:.2f}")[0]
    )
    ax[2].semilogx(OM, np.angle(G_eval, deg=True))
    
    ax[0].plot(G_eval[0].real, G_eval[0].imag, 'x', c=f"C{idx}")
    ax[1].loglog(OM[0], np.abs(G_eval[0]), 'x', c=f"C{idx}")
    ax[2].semilogx(OM[0], np.angle(G_eval[0], deg=True), 'x', c=f"C{idx}")

G_eval = (S**2 + 2.* 1. * om1 * S + om1**2)

ax[0].plot(G_eval.real, G_eval.imag, 'k')
hnd.append(
    ax[1].loglog(OM, np.abs(G_eval), 'k', label = rf"Crit. damp.")[0]
)
ax[2].semilogx(OM, np.angle(G_eval, deg=True), 'k')

ax[0].set(title="Imaginary plane", 
          xlabel="$\mathfrak{Re}\{G(s)\}$", ylabel="$\mathfrak{Im}\{G(s)\}$", 
          xscale='symlog', yscale='symlog')
ax[0].legend(handles=hnd)
ax[1].set(title="Bode plot", ylabel = "$|G(s)|$")
ax[2].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[2].yaxis.set_major_locator(MultipleLocator(90))

display(fig)

# %% [markdown]
# ### Combining transfer function blocks
# We're going to design three filters here to wrap up: a notch filter, a lead-lag filter and a PID! First the notch, it consists of a pole pair and a zero pair with the same natural frequency, but 
# different damping ratios. I'll also stop plotting the complex plane, you get the link by now.

# %%
om_n, zeta_p, zeta_z = 0.7, 5e-1, 1e-4

OM = np.logspace(-1, 1, 400)
S = OM*1j

G_leadlag_eval_z = (S**2 + zeta_z * om_n * S + om_n**2)
G_leadlag_eval_p = 1/ (S**2 + zeta_p * om_n * S + om_n**2)
G_leadlag_eval = G_leadlag_eval_z * G_leadlag_eval_p

fig, ax = plt.subplots(2,1)
ax[0].loglog(OM, np.abs(G_leadlag_eval), 'k')
ax[0].loglog(OM, np.abs(G_leadlag_eval_z), '--')
ax[0].loglog(OM, np.abs(G_leadlag_eval_p), '--')
ax[1].semilogx(OM, np.angle(G_leadlag_eval, deg=True), 'k')
ax[1].semilogx(OM, np.angle(G_leadlag_eval_z, deg=True), '--')
ax[1].semilogx(OM, np.angle(G_leadlag_eval_p, deg=True), '--')

ax[0].set(title="Bode plot - notch filter", ylabel = "$|G(s)|$")
ax[1].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[1].yaxis.set_major_locator(MultipleLocator(90))

display(fig)

# %% [markdown]
# You can figure out yourself what color is the poles/zeros <3. Notice how the curves can be 'added' when in reality they're multiplied, because they're plotted on a log scale.
# 
# Now the lead lag: it also consists of a pole pair and a zero pair, but with different natural frequencies and the same damping ratio, and we need to 
# compensate the steady state gain (for $\omega = 0$):

# %%
om_p, om_z, zeta1 = .1, 4., 0.05
k2 = om_p**2 / om_z**2

OM = np.logspace(-3, 3, 400)
S = OM*1j

G_leadlag_eval_z = (S**2 + zeta1 * om_z * S + om_z**2)
G_leadlag_eval_p = 1/ (S**2 + zeta1 * om_p * S + om_p**2)
G_leadlag_eval = G_leadlag_eval_z * G_leadlag_eval_p * k2

fig, ax = plt.subplots(2,1)
ax[0].loglog(OM, np.abs(G_leadlag_eval), 'k')
ax[0].loglog(OM, np.abs(G_leadlag_eval_z), '--')
ax[0].loglog(OM, np.abs(G_leadlag_eval_p), '--')
ax[1].semilogx(OM, np.angle(G_leadlag_eval, deg=True), 'k')
ax[1].semilogx(OM, np.angle(G_leadlag_eval_z, deg=True), '--')
ax[1].semilogx(OM, np.angle(G_leadlag_eval_p, deg=True), '--')

ax[0].set(title="Bode plot - lead-lag filter", ylabel = "$|G(s)|$")
ax[1].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[1].yaxis.set_major_locator(MultipleLocator(90))

display(fig)

# %% [markdown]
# To summarise these configurations of 2nd order systems:
# 
# ![2nd order systems table](figures/secondOrderTFs.svg)
# 
# Next, one of the simplest (and most effective) controllers is the PID controller (any controller is secretely a filter). But we have to cast the standard expression into blocks we know:
# $$ K_p + \frac{K_i}{s} + K_d s = \frac{K_p s + K_i}{s} + K_d s = \frac{K_d s^2 + K_p s + K_i}{s}.$$
# So that means one pole at 0 and two zeros:
# %%
Kp1, Ki1, Kd1 = .1, 7., 20.
PID1 = (Kd1 * S**2 + Kp1 * S + Ki1) / S

OM = np.logspace(-2, 2, 400)
S = OM*1j

fig, ax = plt.subplots(2,1)
ax[0].loglog(OM, np.abs(PID1), 'k')
ax[1].semilogx(OM, np.angle(PID1, deg=True), 'k')

ax[0].set(title="Bode plot - PID", ylabel = "$|G(s)|$")
ax[1].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[1].yaxis.set_major_locator(MultipleLocator(90))

display(fig)

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Bode Plots"</div>


# %% [markdown]
# ## Introduction to the Nyquist Criterion
# 
# This exploration will be top-down and quite extensive, so we'll start with the definition and work our way down to the nitty gritty of why and how the criterion works.
# 
# ### Definition
# Wikipedia gives this definition of the criterion:
# 
# >Given a Nyquist contour $\Gamma_s$, let $P$ be the number of poles of $L(s)$ encircled by $\Gamma_s$, and $Z$ be the number of zeros of $1+L(s)$ encircled by $\Gamma_s$. Alternatively, and 
# more importantly, if $Z$ is the number of poles of 
# the closed loop system in the right half plane, and $P$ is the number of poles of the open-loop transfer function $L(s)$ in the right half plane, the resultant contour in the $L(s)$-plane, 
# $\Gamma_{L(s)}$, shall encircle (clockwise) the point $(-1+j0)$ $N$ times such that $N = Z - P$.
# 
# That's pretty overwhelming! Let's take it apart:
# - Nyquist contour $\Gamma_s$: this is defined as the clockwise contour encompassing the Right Half of the imaginary Plane (RHP). It is also the collection of all $s$ we'll consider.
# - $L(s)$: the loop transfer function; the product of the plant/process and controller transfer functions, we'll come back to this.
# - $P$: the number of RHP poles of $L(s)$.
# - $Z$: the number of RHP zeros of $1+L(s)$ **and** the number of RHP poles of the closed loop system.
# - $N$: the number of clockwise rotations around $-1+j0$ of $L(\Gamma_s)$, and $N=Z-P$.
# 
# ### Stability
# So $Z$ is supposedly the number of unstable poles of the closed loop system. Then if $Z=0$, there are no RHP poles destabilizing the closed loop, so the closed loop is stable! Now we also 
# got that $N=Z-P\rightarrow Z=N+P$, so we can just count the clockwise encirclements in the Nyquist plot $L(\Gamma_s)$ and add it to the number of RHP poles of $L(s)$ and if that's zero the 
# closed loop is stable! Wait but something added to something should be zero? Good question, $N$ is positive for clockwise encirclements, but negative for counterclockwise encirclements. So, alternatively, 
# we can say the closed loop is stable if (and only if) the **net** number of counterclockwise encirclements of $-1+j0$ is equal to the number of RHP poles of $L(s)$. This is our main use for the Nyquist 
# criterion, because this is how we assess stability.
# 
# ### Where do they get these transfer functions from?
# Another excellent question! This is where it gets a tad more difficult, but we'll soon look at some pictures too. First, we have a quick look at the closed loop transfer function again. Our 
# block diagram with reference $r$, output error $e$, actuation $u$, output $y$, controller $C$, and process $G$ is 
# 
# ![](figures/CLsys.svg)
# 
# Then in the frequency domain the transfer from the reference to the output is $\frac{Y(s)}{R(s)}$. We get this by following the chain as 
# $$Y(s) = G(s)U(s) = G(s)C(s)E(s) = L(s)E(s) = L(s)(R(s)-Y(s))$$
# $$\rightarrow (1+L(s))Y(s) = L(s)R(s) \rightarrow \frac{Y(s)}{R(s)} = \frac{L(s)}{1+L(s)}.$$
# 
# So we can conclude that zeros of $1+L(s)$ are poles of our closed loop system. We saw $1+L(s)$ before! *And they were talking about its zeros!!* In the Nyquist criterion, $Z$ was defined as 
# "the number of zeros of $1+L(s)$ encircled by $\Gamma_s$" **and** "the number of poles of the closed loop system in the right half plane." We've now discovered why these are linked!
# 
# ### Why encirclements then?
# This is where we start letting go of formal mathematics and start going more by visuals. If you're cool however, you can find out more about the formalities in 
# [Cauchy's argument principle](https://en.wikipedia.org/wiki/Argument_principle). It'll take a couple of steps to arrive at the source of the encirclements, but no worries. We'll go through it step by step. 
# 
# ### First the contour
# We'll not go for the full Nyquist contour $\Gamma_s$ immediately, but first take a smaller contour $\Gamma$. Lets define that!


# %%
## Create contour
Q = np.linspace(start=0, stop=2*np.pi, num=100, endpoint=True)
R = 2
cntr = R * np.exp(1j * Q)
cntr[cntr.real < 0] = cntr[cntr.real < 0].imag * 1j

fig1, ax = plt.subplots()
drawContour(ax, cntr)

display(fig1)

# %% [markdown]
# That's a nice semicircle of finite radius! In the end we want to 'walk along the contour' to get its mapping. So we have our complex variable $s$ for our transfer functions and this we'll walk 
# along the contour. That looks like this:

# %%

fig1, ax = plt.subplots()
drawContour(ax, cntr)
sc = ax.scatter([cntr[0].real], [cntr[0].imag], marker='x', color='red')
ax.legend([r'$\Gamma$', r'$s$'])

def animFun(t): 
    sc.set_offsets([cntr[t].real, cntr[t].imag])
    return sc, 
anim = animation.FuncAnimation(fig1, func=animFun, frames=len(Q), interval=30, blit=True)

display(anim)

# %% [markdown]
# ### Shuffling $L(s)$
# Next, we will demonstrate the origin of the encirclements. We need to rewrite $L(s)$ a little first though. For a Linear Time Invariant (LTI) system the loop function can be written as a fraction 
# of two polynomials with the respective roots being the zeros and poles of the loop function. Many words, this is it in maths
# $$ L(s) = \frac{N_L(s)}{D_L(s)} $$ 
# and 
# $$ 1+L(s) = \frac{N_L(s) + D_L(s)}{D_L(s)}.$$
# From this we can see that $L(s)$ and $1+L(s)$ share the same poles, which are the roots of $D_L(s)$.
# 
# Suppose we call $1+L(s) = F(s)$ for now. We are interested in the zeros of $F(s)$, since these are our closed loop poles as we deduced before. Also we know that the sum of two polynomials is another polynomial, 
# so we can write
# $$ F(s) = \frac{N_L(s) + D_L(s)}{D_L(s)} = \frac{N_F(s)}{D_F(s)}.$$ 
# The roots of $N_F(s)$ are therefore zeros of $F(s)$.
# 
# ### Polynomials
# We'll quickly jump into polynomials for a second. Polynomials are largely defined by their roots. For example, this is what we do when we factorize a parabola formula to find its roots. 
# Similar to that, we can write a polynomial $N_F(s)$ as the product of its $Z$ number of roots, $z_k$, as
# $$ N_F(s) = (s - z_1)(s-z_2)...(s-z_Z) = \prod_{k=1}^Z (s - z_k).$$
# Then we can also define $N_F^\dagger (s) = \prod_{k=2}^Z (s - z_k)$, such that 
# $$N_F(s) = (s-z_1)N_F^\dagger (s).$$
# 
# ### Inside or outside
# Now we're going to make two categories: zeros inside $\Gamma$ and zeros outside $\Gamma$. Let's say $z_1$ lies outside $\Gamma$ and look at the behaviour of *the phase of* $(s-z_1)$ when 
# we walk $s$ along $\Gamma$.

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
    ax[1].set(aspect='equal', xlabel="$\mathfrak{Re}\{G(s)\}$", ylabel="$\mathfrak{Im}\{G(s)\}$", xlim=[-rm+np.real(z1), rm+np.real(z1)], ylim=[-rm+np.imag(z1),rm+np.imag(z1)])

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
    return arr, phArc, phl2, r, rm

def animate(t):
    for i in range(2):
        arr[i].xy = (cntr[t].real, cntr[t].imag)
    phArc.theta2 = np.angle(r[t], deg=True)
    phl2[0].set_data([np.real(z1),np.real(z1)+np.real(rm/2*np.exp(1j*np.angle(r[t])))], 
                    [np.imag(z1),np.imag(z1)+np.imag(rm/2*np.exp(1j*np.angle(r[t])))])
    return arr[0], arr[1], phArc, phl2[0]

fig2, ax = plt.subplots(1, 2)
drawContour(ax[0], cntr)

## define the zero
z1 = 1.1*R*np.exp(9/7*np.pi*1j) # You can change me if you'd like

arr, phArc, phl2, r, rm = anim_init(z1, ax)
anim = animation.FuncAnimation(fig2, func=animate, frames=len(Q), interval=50, blit=True)

display(anim)


# %% [markdown]
# Okay, that's looking cool! The angle doesn't seem to be changing all that much, but it's funny to look at I guess. Let's move that zero into $\Gamma$ now and see what happens.

# %%
fig3, ax = plt.subplots(1, 2)
drawContour(ax[0], cntr)

## define the zero
z1 = 0.8*R*np.exp(-2/7*np.pi*1j) # You can change me if you'd like

arr, phArc, phl2, r, rm = anim_init(z1, ax)
anim = animation.FuncAnimation(fig3, func=animate, frames=len(Q), interval=50, blit=True)

display(anim)

# %% [markdown]
# *Wait...* was that a circle? Actually, that makes sense right? $\Gamma$ goes around $z_1$, so then it has to make a full circle. Woah okay, so $(s-z_1)$ makes a full rotation if 
# (and yes, only if) $z_1$ is on the inside of $\Gamma$, otherwise it just oscilates.
# 
# Cool, but this is only for one zero though? What about the others? Well, remember the polynomial
# $$ N_F(s) = \prod_{k=1}^Z (s - z_k)?$$
# It says here that $N_F(s)$ is the product of a bunch of complex numbers. We know what that means, adding the phases! So any zero inside $\Gamma$ adds a full clockwise encirclement 
# to $N_F(\Gamma)$ (and also $F(\Gamma)$) around **the origin**. What do the outside zeros add though? There are two options: if the zero is real, it splits $\Gamma$ exactly in half 
# and doesn't add any phase overall, if the zero is complex it does add some phase of course. However, complex zeros come in conjugate pairs, so these phase additions cancel out because 
# they are mirrored over the real axis.
# 
# How do poles behave here actually? Very similar really, but since the poles are in the denominator of $F(s)$, they subtract phase rather than add. So poles inside $\Gamma$ cause 
# counterclockwise encirclements of the origin.
# 
# ### I forgot what we were trying to figure out
# Yes, me too. Let's run it backwards. We know that zeros inside the contour add a full clockwise rotation and poles add a counterclockwise rotation. If we expand our finite $\Gamma$ 
# to the infinite Nyquist contour $\Gamma_s$, this will encompass all right half plane (RHP) zeros and poles. So the number of clockwise encirclements of the origin of $F(\Gamma_s)$ 
# equals the number of RHP zeros minus the RHP poles. That sounds very familiar! It's similar to the $N=Z-P$ equation, except that is shifted. Realise that
# $$F(\Gamma_s) = 1 + L(\Gamma_s),$$
# so encirclement of $0+j0$ by $F(\Gamma_s)$ means encirclement of $-1 + j0$ by $L(\Gamma_s)$.
# 
# And that's nearly it! Looking back at the practically applicable Nyquist criterion:
# > The closed loop is stable if (and only if) the net number of counterclockwise encirclements of $-1+j0$ is equal to the number of RHP poles of $L(s)$.
# 
# We're missing one last step. Since $L(s)$ and $F(s)$ share the same poles, and the number of counterclockwise encirclements of $-1+j0$ is equal to the number of these poles in the 
# RHP, the number of CCW encirclements and the number of RHP poles of $L(s)$ from the criterion are only equal if there are no RHP zeros of $F(s)$ adding clockwise encirclements. 
# Then, since there are no RHP zeros of $F(s)$, the closed loop is stable!
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
# We've seen both Bode plots and the Nyquist plot now. However, just a recap: Bode plots have $s=j\omega$ on the horizontal axes and on the two vertical axes is the magnitude and 
# phase of the transfer function. We've also been looking at magnitude and phase with Nyquist so we must be able to relate the two! Lets look at a dummy loop transfer function 
# $$L(s) = 150\cdot \frac{s(s-10)}{(s + 1)(s+50)^2}\cdot e^{-10^{-2}s}.$$
# (Determine the poles and zeros for yourself)
# 

# %%
OM = np.logspace(-2, 4, 900)
S = OM*1j

L1 = lambda s :  150* s*(s - 1e1) / ((s + .1) * (s + 50)**2)
L1_eval1 = L1(S)
L1_eval2 = L1(np.flip(-S))

fig = plt.figure(num="Bode - Nyquist relation")
gs = GridSpec(2,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[:, 0], gs[0, 1], gs[1, 1]]]

ax[0].set(title="Nyquist plot", 
          xlabel="$\mathfrak{Re}\{G(s)\}$", ylabel="$\mathfrak{Im}\{G(s)\}$")
ax[1].set(title="Bode plot", ylabel = "$|G(s)|$")
ax[2].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[2].yaxis.set_major_locator(MultipleLocator(90))

drawContour(ax[0], L1_eval1, c='k', ls='-')
drawContour(ax[0], L1_eval2, c='k', ls='--')
ax[0].plot([-1], [0], 'xr')
ax[1].loglog(OM, np.abs(L1_eval1), 'k')
ax[2].semilogx(OM, unwrap_angle(np.angle(L1_eval1, deg=True)), 'k')
display(fig)

# %% [markdown]
# Now **this**, this is something we can animate🌈

# %%
# OM = np.logspace(-2, 5, 50)
# S = OM*1j

L1_eval1 = L1(S)
L1_eval2 = L1(np.flip(-S))

L1_eval = np.append(L1_eval1, L1_eval2)
OM_full = np.append(OM, np.flip(-OM))

ang1, ang2, ang = unwrap_angle(np.angle(L1_eval1, deg=True)), unwrap_angle(np.angle(L1_eval2, deg=True)), unwrap_angle(np.angle(L1_eval, deg=True))

ax[2].cla()
# ax[2].semilogx(np.abs(OM_full), ang, 'k')
ax[2].semilogx(np.abs(OM_full)[:OM_full.size//2], ang[:OM_full.size//2], 'k')
ax[2].semilogx(np.abs(OM_full)[OM_full.size//2:], ang[OM_full.size//2:], 'k--')

arr = ax[0].annotate("", xy=(L1_eval[0].real, L1_eval[0].imag), 
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color='red'))
magL, = ax[1].plot([OM_full[0], OM_full[0]],
                  [np.abs(L1_eval).min(), np.abs(L1_eval[0])],
                  'rx-')
angL, = ax[2].plot([OM_full[0], OM_full[0]],
                  [0, ang1[0]],
                  'rx-')

ss = np.arange(0, OM_full.size, step= OM_full.size//50, dtype=int)
def animate(t):
    t = ss[t]
    arr.xy = (L1_eval[t].real, L1_eval[t].imag)
    magL.set_data([abs(OM_full[t]), abs(OM_full[t])],
                  [np.abs(L1_eval).min(), np.abs(L1_eval[t])])
    angL.set_data([abs(OM_full[t]), abs(OM_full[t])],
                  [0, ang[t]])
    return arr, magL, angL

anim = animation.FuncAnimation(fig, func=animate, frames=50, interval=150, blit=True)

display(anim)

# %% [markdown]
# ## Performance Margins
# As established in block A: *our model always sucks*. Therefore, we might want to know how prone our controller is to destabilizing the system, because our model is wrong. There are three popular measures for 
# this: the gain margin, the phase margin, and the stability margin. They're quite easy:
# - Gain margin $g$: $\argmin_{g}$ such that $g L(s)$ becomes unstable,
# - Phase margin $\phi$: $\argmin_{|\phi|}$ such that $e^{\phi i}L(s)$ becomes unstable (think about how that translates to a time delay!),
# - Stability margin $s_m$: minimal distance between Nyquist plot, $L(\Gamma_s)$, and the point -1.
# 
# You can read the gain and phase margins from a bode plot, but not the stability margin. In this sense a Nyquist plot contains more information than a Bode plot. Anyways, lets plot those margins for the previous 
# loop function.

# %%
OM = np.logspace(-2, 4.5, 700)
S = OM*1j

L1_eval1 = L1(S)
L1_eval2 = L1(np.flip(-S))

L1mag_eval1 = np.abs(L1_eval1)
L1ph_eval1 = np.angle(L1_eval1, deg=True)
L1ph_eval1 = unwrap_angle(L1ph_eval1)
L1ph_eval1 %= 360

fig = plt.figure(num="Bode - Nyquist relation")
gs = GridSpec(2,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[:, 0], gs[0, 1], gs[1, 1]]]

ax[0].set(title="Nyquist plot", aspect='equal',
          xlabel=r"$\mathfrak{Re}\{L(\Gamma_s)\}$", ylabel=r"$\mathfrak{Im}\{L(\Gamma_s)\}$")
ax[1].set(title="Bode plot", ylabel = "$|G(s)|$")
ax[2].set(xlabel = r"$\omega$", ylabel = r"$\angle G(s)$ / ${}^\circ$")
ax[2].yaxis.set_major_locator(MultipleLocator(90))

drawContour(ax[0], L1_eval1, c='k', ls='-')
drawContour(ax[0], L1_eval2, c='k', ls='--')
ax[1].loglog(OM, L1mag_eval1, 'k')
ax[2].semilogx(OM, L1ph_eval1, 'k')


fig2, ax2 = plt.subplots(1,2)
ax2[0].set(title="Nyquist plot - max gain",  aspect='equal',
          xlabel=r"$\mathfrak{Re}\{g_m L(\Gamma_s)\}$", ylabel=r"$\mathfrak{Im}\{g_m L(\Gamma_s)\}$")
ax2[1].set(title="Nyquist plot - max phase",  aspect='equal',
          xlabel=r"$\mathfrak{Re}\{e^{\phi i}L(\Gamma_s)\}$", ylabel=r"$\mathfrak{Im}\{e^{\phi i}L(\Gamma_s)\}$")

[x.plot([-1], [0], 'xr') for x in [ax[0], ax2[0], ax2[1]]]


## Determine gain margin
ax[2].axhline(180., c='C0', ls=':')

idx_g_m = np.abs((L1ph_eval1 - 180.)).argmin()
l1, = ax[2].plot([OM[idx_g_m]], [L1ph_eval1[idx_g_m]], 'o', c='C0', label="Gain Margin")
ax[1].plot([OM[idx_g_m]], [L1mag_eval1[idx_g_m]], 'o', c='C0')
[ax[i].axvline(OM[idx_g_m], c='C0', ls=':') for i in [1, 2]]
g_m = 1/L1mag_eval1[idx_g_m]
[ax2[0].plot(g_m*L1_evalq.real, g_m*L1_evalq.imag, c='C0', ls='-.') for L1_evalq in [L1_eval1, L1_eval2]]

ax[0].plot([0, -L1mag_eval1[idx_g_m]], [0,0], c='C0', ls=':')

print(f"Gain margin = {g_m:1.2f}")

## Determine phase margin
ax[1].axhline(1., c='C1', ls=':')

idx_phi_m_candidates = [idx for idx in range(L1mag_eval1.size - 1) if (L1mag_eval1[idx]-1) * (L1mag_eval1[idx+1]-1) < 0. 
                        and abs(L1ph_eval1[idx]) >= 90. ]
idx_phi_m = idx_phi_m_candidates[0]
l2, = ax[1].plot(OM[idx_phi_m], L1mag_eval1[idx_phi_m], 'o', c='C1', label="Phase Margin")
ax[2].plot(OM[idx_phi_m], L1ph_eval1[idx_phi_m], 'o', c='C1')
[ax[i].axvline(OM[idx_phi_m], c='C1', ls=':') for i in [1, 2]]
phi_m = np.abs(180 + L1ph_eval1[idx_phi_m])
[ax2[1].plot((np.exp(np.deg2rad(phi_m)*1j)*L1_evalq).real, (np.exp(np.deg2rad(phi_m)*1j)*L1_evalq).imag, 
            c='C1', ls='-.') for L1_evalq in [L1_eval1, L1_eval2]]

a = L1mag_eval1[idx_phi_m] * np.exp(np.linspace(0, np.asin(np.sin(phi_m * np.pi / 180.)))*1j)
ax[0].plot(-a.real, a.imag, c='C1', ls=':')
print(f"Phase margin = {np.asin(np.sin(phi_m * np.pi / 180.)) * 180. / np.pi:1.2f} degrees")

## Determine stability margin
idx_s_m = np.abs(L1_eval1 + 1).argmin()
s_m = np.abs(L1_eval1 + 1).min()
l3, = ax[0].plot([-1, L1_eval1[idx_s_m].real],
           [0, L1_eval1[idx_s_m].imag],
           'g-.', label="Stability Margin")

ax[1].legend(handles=[l1,l2,l3])
print(f"Stability margin = {s_m:1.2f}")

display(fig, fig2)

# %% [markdown]
# 
# ## Minimum Phase Systems
# Bode has one last curve-ball for you lovely people, he has a relation named after him that couples the phase and gain of TFs for so-called "Minimum Phase Systems" (MPS). We'll look into that more in a bit, 
# first Bode's relation. Be warned, it's not pretty at first sight:
# 
# TODO: I'm confused man, the stuff in the slides doesn't make sense to me...
# 
# Very briefly, a system is minimum phase iff:
# - There are no RHP zeros *and*
# - There are no time delays.

# %% [markdown]
# ### My Bode's Relation confusion
# The relation is defined in the slides (of '24/'25) as
# $$ \angle G(i\Omega) = \frac{1}{\pi}\int_{-\infty}^{\infty}\frac{\text{d}\log|G(i\omega)|}{\text{d}\log(\omega)}\;\log\left|\frac{\omega+\Omega}{\omega-\Omega}\right|
# \frac{\text{d}\omega}{\omega}.$$
# Now my issue is that the integral is from $-\infty$ onwards, but that $\log(x)$ is not defined for $x\leq0$, and there are many logarithms. Investigating 
# the three factors one by one:
# 
# $\frac{\text{d}\log|G(i\omega)|}{\text{d}\log(\omega)}$ is a derivative. As we've seen before, this is just the slope of the log-log graph for positive frequencies and for negative
# frequencies it's the same but negative. I do have some issues with the notation of the denominator though.
# 
# $\log\left|\frac{\omega+\Omega}{\omega-\Omega}\right|$ is a bit weirder, but logarithms of an absolute value, so we're good for negative values. Let's do some simplifications
# $$ \log\left|\frac{\omega+\Omega}{\omega-\Omega}\right| = \log\frac{|\omega+\Omega|}{|\omega-\Omega|} = \log|\omega+\Omega| - \log|\omega-\Omega|.$$
# Okay, so the function is not defined for $\omega = \Omega$.
# 
# Lastly, there is $\frac{1}{\omega}$, causing problems when $\omega=0$.
# 
# ### So do better then
# Yeah, I'll try. I'm at my parents at the moment and don't have my books, so we'll all have to make do with wikipedia.
# 
# In [this paragraph](https://en.wikipedia.org/wiki/Minimum_phase#Relationship_of_magnitude_response_to_phase_response), they define the relation through the Hilbert transform, 
# $\mathcal{H}$. The relation is then
# $$ \angle G(s) = -\mathcal{H}\{\log|G(s)|\}, $$ 
# with the inverse 
# $$ \log|G(s)| = \log|G(i\infty)| + \mathcal{H}\{\angle G(s)\} .$$
# Here, the Hilbert transform is defined in time-domain as 
# $$ \mathcal{H}\{x(t)\} =  \frac{1}{\pi}\int_{-\infty}^{\infty} \frac{x(\tau)}{t-\tau}\text{d}\tau$$
# [This page](https://en.wikipedia.org/wiki/Hilbert_transform#Relationship_with_the_Fourier_transform) also exists. Also see page 172 of Skogestad. This is a bigger time investment than I expected.
# 
# I give up (for now)


# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Frequency Domain Analysis"</div>

# %% [markdown]
# # PID control
# **So I'm switching around this lecture and the next, because that makes more sense to me. I'll cover this one after the next.** I believe the course would benefit from a bigger contrast between system analysis 
# and controller synthesis. 
# <div style="text-align:center;background-color:tomato;">End of lecture "PID control"</div>

# %% [markdown]
# # You will love the frequency domain after this, I promise
# Looking back at the general controller architecture:
# ![General feedback loop](figures/CLsys_general.svg)
# 
# Then the complete transfer function collection is
# $$ \begin{bmatrix} y \\ \eta \\ v \\ u \\ e \end{bmatrix} = \frac{1}{1+PC}
# \begin{bmatrix} 
# PCF   & P     & 1     \\
# PCF   & P     & -PC   \\
# CF    & 1     & -C    \\
# CF    & -PC   & -C    \\
# F     & -P    & -1    
# \end{bmatrix} \begin{bmatrix} r \\ d \\ n \end{bmatrix} \triangleq
# \begin{bmatrix} 
# TF   & PS     & S     \\
# TF   & PS     & -T   \\
# CFS  & S      & -CS    \\
# CFS  & -T     & -CS    \\
# FS   & -PS    & -S    
# \end{bmatrix} \begin{bmatrix} r \\ d \\ n \end{bmatrix} ,$$
# where $S=\frac{1}{1+PC}=\frac{1}{1+L}$ is called the sensitivity function and $T=\frac{PC}{1+PC}=\frac{L}{1+L}=SL$ is called the complementary sensitivity function. They're called complements because
# $$ S + T = \frac{1}{1+PC} + \frac{PC}{1+PC} = I.$$
# 
# ## What about the performance?
# Fun fact: you can derive the effectiveness of feedback control from that transfer matrix. Looking at the last row which represents the transfers from the reference and disturbances
# to the tracking error, $e$. The smaller this error, the better the feedback control is performing. Now also note that every element in this last row is right multiplied with $S$, meaning that smaller magnitudes
# of $S$ result in smaller tracking errors. So
# - $|S(s)| < 1 \rightarrow $ disturbances are attenuated/rejected (this is good),
# - $|S(s)| = 1 \rightarrow $ equivalent to open loop control (meh),
# - $|S(s)| > 1 \rightarrow $ disturbances are amplified (this is bad).
# 
# We've looked at controller performance before with the gain, phase, and stability margins. Of course these are still valid and we'll build further on this. Just to reframe that theory, we have to define
# the cross-over frequencies $\omega_\text{gc}\leftarrow|L(\omega_\text{gc})|=1$ and $\omega_\text{pc}\leftarrow\angle L(\omega_\text{pc})=\pm 180^\circ$ for the gain and phase cross-overs (you know $L=PC$).
# Then the gain margin GM $=1/|L(\omega_\text{pc})|$ and phase margin PM $=180^\circ - |\angle L(\omega_\text{gc})|$.
# 
# With our new theory we can also define the closed loop bandwidth, $\omega_\text{B}$, of our controlled system! This is a measure of until what frequency we can reject disturbances and it's defined as the 
# frequency where $|T(s)|$ first crosses $\frac{1}{\sqrt2}\approx0.707\approx -3$ dB from above. Good, we have the frequencies of interest for the gain and phase margins.
# 
# Now for the stability margin, $s_m$, that we defined earlier as the minimal distance between $L(s)$ and the point -1. Equivalently, this is the minimum of $|1 + L(s)|$, which coincidentally is the denominator of $S$. 
# Therefore the stability margin occurs when the sensitivity function magnitude is the largest. Utilising that property, we define
# $$ M_S = \max |S(s)| = \frac{1}{s_m} .$$
# Now we have to perform some geometry magic, but to explain that we need an exampe Nyquist plot

# %%
L2 = lambda s :  1.2 / (s+.7)**3

OM = np.logspace(-2, 2, 700)
S = OM*1j

L2_eval1 = L2(S)
L2_eval2 = L2(np.flip(-S))

L2mag_eval1 = np.abs(L2_eval1)
L2ph_eval1 = unwrap_angle(np.angle(L2_eval1, deg=True))
L2ph_eval1 %= 360

fig, ax = plt.subplots(num="Performance")

drawContour(ax, L2_eval1, c='k', ls='-')
drawContour(ax, L2_eval2, c='k', ls='--')

ax.set(title="Nyquist plot", aspect='equal',
          xlabel=r"$\mathfrak{Re}\{L(\Gamma_s)\}$", ylabel=r"$\mathfrak{Im}\{L(\Gamma_s)\}$")

## Determine gain margin
idx_g_m = np.abs((L2ph_eval1 - 180.)).argmin()
g_m = 1/L2mag_eval1[idx_g_m]
l1, = ax.plot([0, -L2mag_eval1[idx_g_m]], [0,0], c='C0', ls=':', label="1/GM")

print(f"Gain margin = {g_m:1.2f}")

## Determine phase margin
idx_phi_m_candidates = [idx for idx in range(L2mag_eval1.size - 1) if (L2mag_eval1[idx]-1) * (L2mag_eval1[idx+1]-1) < 0. 
                        and abs(L2ph_eval1[idx]) >= 90. ]
idx_phi_m = idx_phi_m_candidates[0]
phi_m = np.abs(180 + L2ph_eval1[idx_phi_m])
a = L2mag_eval1[idx_phi_m] * np.exp(np.linspace(0, np.asin(np.sin(phi_m * np.pi / 180.)))*1j)
l2, = ax.plot(-a.real, abs(a.imag), c='C1', ls=':', label="PM")
print(f"Phase margin = {np.asin(np.sin(phi_m * np.pi / 180.)) * 180. / np.pi:1.2f} degrees")

## Determine stability margin
idx_s_m = np.abs(L2_eval1 + 1).argmin()
s_m = np.abs(L2_eval1 + 1).min()
l3, = ax.plot([-1, L2_eval1[idx_s_m].real],
           [0, abs(L2_eval1[idx_s_m].imag)],
           'g-.', label="Stability Margin")
SMcirc = s_m * np.exp(np.linspace(0,2*np.pi) * 1j)
ax.plot(SMcirc.real - 1, SMcirc.imag,
           'g-.', label="Stability Margin")

ax.legend(handles=[l1,l2,l3])
print(f"Stability margin = {s_m:1.2f}")

display(fig)

# %% [markdown]
# So lets have a gander at what this plot tells us, starting with the gain margin. We can say for sure that the Nyquist plot on the real axis is *at least* as far away as $s_m$, 
# otherwise that would be the stability margin. So we have $s_m \leq 1 - \frac{1}{\text{GM}}$ from that statement, and some rewriting gives
# $$ s_m \leq 1 - \frac{1}{\text{GM}} \rightarrow 1 - s_m \geq \frac{1}{\text{GM}} \rightarrow \text{GM} \geq \frac{1}{1 - s_m} .$$
# Coupling back to $M_S$:
# $$ \frac{1}{1 - s_m} = \frac{M_S}{M_S - 1} \leq \text{GM}.$$
# 
# Now the phase margin is a tad more annoying, because it actually involves triangles and we need some extra lines. (Stolen from: Skogestad p.35)
# 
# 

# %%
PM = L2_eval1[idx_phi_m].real + abs(L2_eval1[idx_phi_m].imag) * 1j
SM = L2_eval1[idx_s_m].real + abs(L2_eval1[idx_s_m].imag) * 1j

# Redraw figure and zoom to relevant part
fig, ax = plt.subplots(num="Performance - Phase Margin bounds")

ax.set(title="Nyquist plot", aspect='equal',
          xlabel=r"$\mathfrak{Re}\{L(\Gamma_s)\}$", ylabel=r"$\mathfrak{Im}\{L(\Gamma_s)\}$")

ax.plot(L2_eval1.real, L2_eval1.imag, c='k', ls='-', alpha=.4)
ax.plot(L2_eval2.real, L2_eval2.imag, c='k', ls='--', alpha=.4)
ax.plot(-a.real/abs(a), abs(a.imag)/abs(a), c='C1', ls=':', label="PM")
ax.plot([-1, SM.real],
        [0, SM.imag],
        'g-.', label="Stability Margin")
SMcirc = s_m * np.exp(np.linspace(0,2*np.pi) * 1j)
ax.plot(SMcirc.real - 1, SMcirc.imag,
           'g-.', label="Stability Margin")
ax.set(xlim=[-1 - s_m*1.2, .1], ylim=[-.1, PM.imag+.1])

# Draw construction lines and points
ax.plot([0, -1, PM.real, 0, (PM.real - 1) /2],
        [0,  0, PM.imag, 0, (PM.imag    ) /2],
        'k')
l = [ax.plot([p.real], [p.imag], 'o', label=n)[0] for p, n in zip([PM, (PM - 1)/2], ['$P$', '$Q$'])]
ax.legend(handles=l)
display(fig)

# %% [markdown]
# What does this triangle whisper in the night? Well first of all that $P = L(i\omega_\text{gc})$ and $Q = \frac12(P - 1)$, and per definition $\angle (-1,0,P) = $ PM. From similarity you 
# also get the bisection property $\angle (-1,0,Q) = \frac12$ PM and note that $\angle (P,Q,0)$ is a right angle. Then, the length of $(-1, P)$ is 
# $$|P+1|=|L(i\omega_\text{gc})+1|=\frac{1}{|S(i\omega_\text{gc})|}=2\sin(\frac12\text{PM}) \rightarrow |S(i\omega_\text{gc})| = \frac{1}{2\sin(\frac12\text{PM})}$$
# Now as a last step, per definition of the stability margin $|P+1|\geq s_m$. Therefore, $2\sin(\frac12\text{PM}) \geq s_m$ and $M_S \leq 
# \frac{1}{2\sin(\frac12\text{PM})}$. Isolating PM gives
# $$PM \geq 2\arcsin(\frac{1}{2 M_S}).$$
# DONE. Now if we get $M_S=2$, we are guaranteed to have a gain margin more than 2 and a phase margin more than $29^\circ$, guaranteeing good performance.
# 
# # Loop Shaping
# We've done a lot of system analysis up to now, but it is time for some controller synthesis! To design a controller, one can do something called loop shaping. To be honest, this is easier said than done, but 
# we already have the theory. Firstly, we determine the natural crossover frequency $\omega_\text{gc}$ from $L\mid_{C=1}$. Then we say we have three regions: low frequencies, the crossover region around 
# $\omega_\text{gc}$, and high frequencies. Now the trick is to add poles and zeros (pairs) to $C$, such that
# 
# | Frequencies | $C(s)$  | $\mid S(s)\mid$ |
# | --------    | ------- | --------        |
# | Low         | High gain ensures unscaled tracking       | Low gain ensures disturbance rejection            |
# | Cross-over  | Ensure good GM and PM     | Ensure good SM             |
# | High        | Low gain ensures measurement noise rejection      | -            |
# 
# That unscaled tracking cell confused me slightly, so briefly why: the transfer function $G_{yr} = TF = \frac{L}{1+L}F$. Ignoring $F$ for a second, $\lim_{|L|\rightarrow\infty}\frac{L}{1+L} = 1$, meaning there 
# is unitary gain for constant input, i.e. unscaled reference tracking.
# 
# ## Easier said...
# So lets run an example and you'll understand better (I hope). This will be very much a Plato-style of explanation (coincidentally my favourite style). We start with a unitary 
# feedback controller and see what plant we're dealing with.
# 
# %%
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({ 'mathtext.fontset':         'cm',
                      'font.size':          12.0,               'axes.labelsize':           'medium',
                      'xtick.labelsize':    'x-small',          'ytick.labelsize':          'x-small',
                      'axes.grid':          True,               'axes.formatter.limits':    [-3, 6],
                      'grid.alpha':         0.5,                'figure.figsize':           [11.0, 4],
                      'figure.constrained_layout.use': True,    'scatter.marker':           'x',
                      'animation.html':     'jshtml'
                    })

from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from IPython.display import display, Markdown

import warnings
warnings.filterwarnings("ignore")

import control as cm
from helperFunctions import *

###############################################
SYS = loopShaper()
fig = plt.figure(figsize=[15, 8])
gs = GridSpec(4,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[0,0], gs[1, 0], gs[2, 0], gs[:3, 1], gs[3,:]]]

SYS.plot_LS(ax)
display(fig)

# %% [markdown]
# What should you get from this plot? Well, lets start with the crossover frequencies: they're around 30 or 300 rad/s. The loop transfers below and above have a low gain. Our margins are looking good however 
# and we can gain some performance for sure. Lets work low to high frequencies. For the lower frequencies our sensitivity needs low gains, so our loop needs high gains, also ensuring good tracking. 
# It doesn't, so lets add an integrator:

# %%
SYS.Cpoles = [0]
SYS.OM = np.logspace(-5, 5, 1000)

[a.cla() for a in ax]
SYS.plot_LS(ax)
display(fig)

# %% [markdown]
# We need a bit more downslope around 1E-3 rad/s, i.e. more poles:

# %%
SYS.Cpoles = [0, -1e-3]

[a.cla() for a in ax]
SYS.plot_LS(ax)
display(fig)

# %% [markdown]
# Okaayyy, lower frequencies looking good. Zooming in on the crossover frequency next:

# %%
[a.set_xlim([1e-2, 1e2]) for a in ax[:3]]
ax[0].set_ylim([1e-3, 5e1])
display(fig)

# %% [markdown]
# Notes: our PM is very large, as well as our GM. In this case we can decrease both of these with the gain. Lets aim for $35^\circ$ PM, looking at the phase plot that happens at 50 or so rad/s, the 
# magnitude there now is $5E-4$. *Therefore*, our gain can be $2E3$:

# %%
SYS.Cgain = 2e2

[a.cla() for a in ax]
SYS.plot_LS(ax)
display(fig, np.pi/3)

# %% [markdown]
# Honestly I suck at loop shaping.
# 
# ## Feedforward control
# We've been ignoring the feedforward block up to now, but really it's very powerful when you can measure the output noise. 
# Consider when there's no input noise $d$, which is often the case, and we add an extra feedforward control $u_\text{ff}$ 
# such that $u = Ce + u_\text{ff}$. Now append/recall $y = TFr + PSu_\text{ff} + Sn$
# Then with the measurement of the output noise $n$, define $u_\text{ff} = P^{-1}Fr - P^{-1}n$
# $$ y = (1- S)Fr + PSu_\text{ff} + Sn  = Fr - SFr + PS(P^{-1}Fr - P^{-1}) + Sn = Fr - SFr + PSP^{-1}Fr - PSP^{-1}n + Sn $$
# $$ = Fr - SFr + SFr - Sn + Sn = Fr.$$
# This means you have perfect tracking. However, inverting the plant might not always be possible (non minimum phase systems for example). 
# There are more tricks like this when you have knowledge of your disturbances.

# %% [markdown]
# # Fundamental Limitations
# ## Waterbed effect
# The funny thing about control is, the most important proofs are the proofs that say that something 
# is **impossible**. These are the bounds you cannot circumvent. Bode's integral formula, otherwise known as the waterbed effedct,
#  is one of these. It is defined as: assume L has relative degree $geq$ 2 and $N_p$ RHP-poles $p_i$, then
# $$ \int_0^\infty\log|S(i\omega)|\text{d}\omega = \pi\sum_{i=1}^{N_p} \mathfrak{Re}(p_i).$$
# This is somewhat reminiscent of the Nyquist plot. This integral is basically the integral of the vector going from -1 to the Nyquist plot. 
# Iff this integral makes a half ellips on the onesided Nyquist contour, there is an encirclement, there is a RHP-pole.
# 
# What it's also saying is: you can never change the value of that integral (assuming you don't add RHP poles). Therefore, if you change C 
# to take some magnitude in S away, it has to come back at other frequencies.
# 
# ## Time delays
# Time delays are annoying, because they put a hard limit on your bandwidth. Why? Well, a time delay $e^{-\theta s}$, has an idealized $T(s)=e^{-\theta s}$. 
#  The bandwidth is defined as $|T(i\omega_\text{B})| = \frac{1}{\sqrt2}$. Also 
# $$|S + T| = 1 \leq |S| + |T| \rightarrow |S(i\omega_\text{B})| + \frac{1}{\sqrt2} \geq 1 \rightarrow |S(i\omega_\text{B})|  \geq 1 - \frac12\sqrt2$$
# $$ S = 1 - e^{-\theta s} = 1 - \cos(-\theta\omega) - i\sin(-\theta\omega) \rightarrow |S| = \sqrt{ (1 - \cos(-\theta\omega)^2 +  \sin(-\theta\omega)^2}$$
# $$ |S| = 1 \rightarrow  (1 - \cos(-\theta\omega)^2 +  \sin(-\theta\omega)^2 = 1$$
# $$ (1 - \cos(-\theta\omega)^2 +  \sin(-\theta\omega)^2 = 1 = 1 - 2\cos(-\theta\omega) + \cos(-\theta\omega)^2 +  \sin(-\theta\omega)^2 $$
# $$ = 1 - 2\cos(-\theta\omega) + 1 = 1 \rightarrow 1 - 2\cos(-\theta\omega) = 0 \rightarrow \cos(-\theta\omega) = \frac12 \rightarrow -\theta\omega = 
# \pm\frac13\pi \approx \pm 1 \rightarrow \omega \approx \frac{1}{\theta} $$

# %%
w = np.logspace(-2, 2, 800)
fig = plt.figure()
plt.loglog(w, np.abs(1 - np.exp(-1 * w * 1j)))
plt.loglog(w, np.abs( np.exp(-1 * w * 1j)))
display(fig)

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Frequency Domain Design I & II"</div>
# # Closing remark
# I'm so sorry, but everything we've done is, technically speaking, bachelor level control engineering.
