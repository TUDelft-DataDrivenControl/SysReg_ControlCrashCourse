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
from IPython.display import display, Markdown

import warnings
warnings.filterwarnings("ignore")

import control as cm
from helperFunctions import *


# %% [markdown]
# ## Performance Margins
# As established in block A: *our model always sucks*. Therefore, we might want to know how prone our controller is to destabilizing the system, because our model is wrong. There are three popular measures for 
# this: the gain margin, the phase margin, and the stability margin. They're quite easy:
# - Gain margin $g$: $\argmin_{g}$ such that $g L(s)$ becomes unstable,
# - Phase margin $\phi$: $\argmin_{|\phi|}$ such that $e^{\phi i}L(s)$ becomes unstable (think about how that translates to a time delay!),
# - Stability margin $s_m$: minimal distance between Nyquist plot, $L(\Gamma_s)$, and the point -1.
# 
# You can read the gain and phase margins from a bode plot, but not the stability margin. In this sense a Nyquist plot contains more information than a Bode plot. Anyways, lets plot those margins for a 
# loop function.

# %%
OM = np.logspace(-2, 4.5, 700)
S = OM*1j

L1 = lambda s :  200* s*(s - 1e1) / ((s + .1) * (s + 50)**2)

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
# $$ S + T = \frac{1}{1+PC} + \frac{PC}{1+PC} = 1.$$
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
# the cross-over frequencies $\omega_\text{gc}\leftarrow|L(\omega_\text{gc})|=1$ and $\omega_\text{pc}\leftarrow\angle L(\omega_\text{pc})=\pm 180^\circ$ for the gain and phase cross-overs.
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