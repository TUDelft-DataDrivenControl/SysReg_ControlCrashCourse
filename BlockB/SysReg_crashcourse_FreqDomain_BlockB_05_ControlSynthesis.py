# %%
%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

from helperFunctions import *
setPlotStyle()


# %% [markdown]
# # PID control
# **So I'm switching around this lecture and the next, because that makes more sense to me. I'll cover this one after the next.** I believe the course would benefit from a bigger contrast between system analysis and controller synthesis. 
# <div style="text-align:center;background-color:tomato;">End of lecture "PID control"</div>

# %% [markdown]
# # Loop Shaping
# We've done a lot of system analysis up to now, but it is time for some controller synthesis! To design a controller, one can do something called loop shaping. To be honest, this is easier said than done, but we already have the theory. Firstly, we determine the natural crossover frequency $\omega_\text{gc}$ from $L\mid_{C=1}$. Then we say we have three regions: low frequencies, the crossover region around $\omega_\text{gc}$, and high frequencies. Now the trick is to add poles and zeros (pairs) to $C$, such that
# 
# | Frequencies | $C(s)$  | $\mid S(s)\mid$ |
# | --------    | ------- | --------        |
# | Low         | High gain ensures unscaled tracking       | Low gain ensures disturbance rejection            |
# | Cross-over  | Ensure good GM and PM     | Ensure good SM             |
# | High        | Low gain ensures measurement noise rejection      | -            |
# 
# That unscaled tracking cell confused me slightly, so briefly why: the transfer function $G_{yr} = TF = \frac{L}{1+L}F$. Ignoring $F$ for a second, $\lim_{|L|\rightarrow\infty}\frac{L}{1+L} = 1$, meaning there is unitary gain for constant input, i.e. unscaled reference tracking.
# 
# ## Easier said...
# So lets run an example and you'll understand better (I hope). We start with a unitary feedback controller and see what plant we're dealing with.
# 
# %%
SYS = loopShaper()
fig = plt.figure(figsize=[15, 8])
gs = GridSpec(4,2, figure=fig)
ax = [fig.add_subplot(a) for a in [gs[0,0], gs[1, 0], gs[2, 0], gs[:3, 1], gs[3,:]]]

SYS.plot_LS(ax)
display(fig)

# %% [markdown]
# What should you get from this plot? Well, lets start with the crossover frequencies: they're around 30 or 300 rad/s. The loop transfers below and above have a low gain. Our margins are looking good however and we can gain some performance for sure. Lets work low to high frequencies. For the lower frequencies our sensitivity needs low gains, so our loop needs high gains, also ensuring good tracking. It doesn't, so lets add an integrator:

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
ax[0].set_ylim([1e-4, 5e1])
display(fig)

# %% [markdown]
# Notes: our PM is not super big, GM is massive however. In this case we can fix both of these with the gain. Lets aim for $35^\circ$ PM, looking at the phase plot that happens at 50 or so rad/s, the magnitude there now is $5E-4$. *Therefore*, our gain can be $2E3$:

# %%
SYS.Cgain = 2e3

[a.cla() for a in ax]
SYS.plot_LS(ax)
display(fig)

# %% [markdown]
# That's a bit too much overshoot to my liking, and our stability margin is a little too small: so let's pull back a little on that gain to increase the PM (generally reduces overshoot (not always)) and increase the stability margin.

# %%
SYS.Cgain = 1.25e3

[a.cla() for a in ax]
SYS.plot_LS(ax)
display(fig)

# %% [markdown]
# Now that's a nice looking controller!
# 
# ## Feedforward control
# We've been ignoring the feedforward block up to now, but really it's very powerful when you can measure the output noise. Consider when there's no input noise $d$, which is often the case, and we add an extra feedforward control $u_\text{ff}$ such that $u = Ce + u_\text{ff}$. Now append/recall $y = TFr + PSu_\text{ff} + Sn$ Then with the measurement of the output noise $n$, define $u_\text{ff} = P^{-1}Fr - P^{-1}n$
# $$ y = (1- S)Fr + PSu_\text{ff} + Sn  = Fr - SFr + PS(P^{-1}Fr - P^{-1}) + Sn = Fr - SFr + PSP^{-1}Fr - PSP^{-1}n + Sn $$
# $$ = Fr - SFr + SFr - Sn + Sn = Fr.$$
# This means you have perfect tracking. However, inverting the plant might not always be possible (non minimum phase systems for example). There are more tricks like this when you have knowledge of your disturbances.

# %% [markdown]
# # Fundamental Limitations
# ## Waterbed effect
# The funny thing about control is, the most important proofs are the proofs that say that something is **impossible**. These are the bounds you cannot circumvent. Bode's integral formula, otherwise known as the waterbed effect, is one of these. It is defined as: assume L has relative degree $\geq$ 2 and $N_p$ RHP-poles $p_i$, then
# $$ \int_0^\infty\log|S(i\omega)|\text{d}\omega = \pi\sum_{i=1}^{N_p} \mathfrak{Re}(p_i).$$
# This is somewhat reminiscent of the Nyquist plot. This integral is basically the integral of the vector going from -1 to the Nyquist plot. Iff this integral makes a half ellips on the onesided Nyquist contour, there is an encirclement, there is a RHP-pole.
# 
# What it's also saying is: you can never change the value of that integral (assuming you don't add RHP poles). Therefore, if you change C to take some magnitude in S away, it has to come back at other frequencies.
# 
# ## Time delays
# Time delays are annoying, because they put a hard limit on your bandwidth. Why? Well, I really tried to explain it nicely, but to be honest it's just hard. Therefore I'm going for the most boring option and I'm going to paraphrase Skogestad Ch. 5.5 and 5.6. Reading that is honestly better.
# 
# For any plant with a time delay $\theta \rightarrow e^{-\theta s}$, the perfect controller can be at most as fast as the time delay. After all, the perfect control will only affect the output after the time delay. Therefore, an absolute perfect control is limited to a complementary sensitivity function of $T(s)=e^{-\theta s}$. The ideal sensitivity function is therefore $S(s) = 1 - T(s)$. Then a lot of maths,
# $$ S = 1 - e^{-\theta s} = 1 - \cos(-\theta\omega) - i\sin(-\theta\omega) \rightarrow |S| = \sqrt{ (1 - \cos(-\theta\omega))^2 +  \sin(-\theta\omega)^2}$$
# $$ |S(i\Omega)| = 1 \rightarrow  (1 - \cos(-\theta\Omega))^2 +  \sin(-\theta\Omega)^2 = 1$$
# $$  = 1 - 2\cos(-\theta\Omega) + \cos(-\theta\Omega)^2 +  \sin(-\theta\Omega)^2  = 1 - 2\cos(-\theta\Omega) + 1 = 1 \rightarrow 1 - 2\cos(-\theta\Omega) = 0 $$
# $$ \rightarrow \cos(-\theta\Omega) = \frac12 \rightarrow -\theta\Omega = \pm\frac13\pi \approx \pm 1 \rightarrow \Omega \approx \frac{1}{\theta} $$
# Also we need to realise what the perfect sensitivity function is: $|S| = |L^{-1}T|$ (I would love to be able to explain why). As we saw above: $|T|=1$, so $|S| = \frac{1}{|L|}$. That means that $|S(i\Omega)| = 1 \rightarrow |L(i\Omega)| = 1$ and therefore $\Omega$ is the crossover frequency. Now your controller will never be absolutely perfect, because we live in the real world. So your crossover frequency will be lower than this value, making it an upper bound on your crossover frequency and controller speed. Now the slides say delays give a limit on your bandwidth and I'm not sure why.
# ## RHP zeros
# Read Skogestad 5.6, I'm sorry.
# 
# Now just to show *something*, here's $S = 1 - e^{-\theta s},\;\theta=5$
# %%
w = np.logspace(-2, 1, 800)
fig = plt.figure()
plt.loglog(w, np.abs(1 - np.exp(-5 * w * 1j)))
plt.gca().axvline(1/5., color='k', ls='--')
plt.gca().axhline(1., color='k', ls='--')
plt.gca().set(xlabel="$\omega$", ylabel="$|S(i\omega)|$",xlim=[w[0],w[-1]])

display(fig)

# %% [markdown]
# <div style="text-align:center;background-color:tomato;">End of lecture "Frequency Domain Design I & II"</div>
# 
# # I promised PID
# **Note: PID is a regulator, not a tracker, apply it to the error signal.**
# 
# So in the slides there is some info on time-domain tuning of PID, mostly [Ziegler-Nichols](https://en.wikipedia.org/wiki/Ziegler%E2%80%93Nichols_method) methods. Generally they're okay, not very impressive though and almost never optimal. I'm here to show you that PID has loop shaping equivalents. Starting with the standard ideal PID controller:
# $$ C(s) = K_p + \frac{K_i}{s} + sK_d. $$
# Rewriting gives
# $$ C(s) = K_p\frac{s}{s} + \frac{K_i}{s} + sK_d\frac{s}{s} = \frac{K_d s^2 + K_p s + K_i}{s} = K_d \frac{s^2 + \frac{K_p}{K_d} s + \frac{K_i}{K_d}}{s}. $$
# Now we recognise a zero pair and integrator, and control gain $K_d$. Retrieving the damping ratio and natural frequency of this zeros pair gives
# $$\left.\begin{aligned} &2\zeta\omega_0 = \frac{K_p}{K_d} \\ &\omega_0^2 = \frac{K_i}{K_d} \end{aligned} \right\}\therefore (\omega_0, \zeta) = \left(\sqrt{\frac{K_i}{K_d}}, \frac{K_p\sqrt{\frac{K_i}{K_d}}}{2K_i}\right).$$
# And there you have the equivalence between loop shaping and PID tuning. We have
# | Controller | Tools |
# | ---------- | ----- |
# | PID         | Gain, zero pair, integrator |
# | PI          | Gain, one zero, integrator |
# | PD          | Gain, one zero |
# 
# So PID is just a limited version of loopshaping.
# 
# # Closing remark
# I'm so sorry, but everything we've done is, technically speaking, bachelor level control engineering.
