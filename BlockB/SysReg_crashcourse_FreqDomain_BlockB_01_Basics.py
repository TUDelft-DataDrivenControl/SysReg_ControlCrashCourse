# %% [markdown]
# # Frequency Domain
# Okay now stuff actually gets a little tricky so I'll have to be a bit more serious sadly.
# ## The Laplace (and Fourier) Transform is a godsent
# ### Transform of arbitrary signals
# Suppose you have a time-varying signal $u(t)$, then there exists its (bilateral) Laplace or Fourier transform $U(s)$, where $s = \sigma + i\omega$ for the Laplace transform and $s = i\omega$ for the Fourier transform.
# 
# These transforms are defined as 
# $$\mathcal{L}\{u(t)\}(s) = U(s) = \int_{-\infty}^\infty u(t) e^{-st} \text{d}t,\; s \in \mathbb{C}$$
# and
# $$\mathcal{F}\{u(t)\}(\omega) = U(\omega) = \int_{-\infty}^\infty u(t) e^{-i\omega t} \text{d}t,\; \omega \in \mathbb{R}.$$
# And now you know why it's called the frequency domain: $\mathfrak{Re}(e^{-st})$ is a (dampened) oscillation! To be completely honest, this is *way* too big of a subject for me to explain. However, I don't have to because of legends like 3Blue1Brown, go watch these if you want to know more:
# - [watch this video on what Fourier transforms are](https://www.youtube.com/watch?v=r6sGWTCMz2k)
# - [and this one on Laplace transforms.](https://www.youtube.com/watch?v=FE-hM1kRK4Y)
# - [Just watch his playlist on differential equations really, the man's a treasure.](https://www.youtube.com/playlist?list=PLZHQObOWTQDNPOjrT6KVlfJuKtYTftqH6)
# 
# But if you don't care about all that and just want to apply it, you need these key fact:
# - The Fourier transform is a restricted Laplace transform, but most things work for both so I'll just continue with the Laplace transform for now,
# - These transforms basically say that **any** time signal is a sum of (dampened) sinusoids,
# - Both transforms are linear.
# 
# Furthermore, some interesting mathematical fact: starting with the definition again
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
