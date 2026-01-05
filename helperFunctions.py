import numpy as np

def RK4(x, dx, h):
    k1 = dx(x)
    k2 = dx(x+h*k1/2)
    k3 = dx(x+h*k2/2)
    k4 = dx(x+h*k3)
    xp = x + h/6*(k1 + 2*k2 +2*k3 + k4)
    return xp


def vector_field(A, v1, v2, ax, n=8):
    X = np.linspace(-1, 1, num=n, endpoint=True)
    Y = np.linspace(-1, 1, num=n, endpoint=True)

    U = np.empty([n, n])
    V = np.empty([n, n])

    for x, idx1 in zip(X, range(X.size)):
        for y, idx2 in zip(Y, range(Y.size)):
            dx = A@np.array([[x],[y]])
            U[idx2, idx1] = dx[0, 0]
            V[idx2, idx1] = dx[1, 0]
    
    ax.quiver(X, Y, U, V, color='tab:gray')
    l1, =ax.plot([0,  20*v1[0,0]], [0,  20*v1[1,0]], 'k', label="Eig.vec. one")
    l2, =ax.plot([0,  20*v2[0,0]], [0,  20*v2[1,0]], 'k--', label="Eig.vec. two")
    _,  =ax.plot([0, -20*v1[0,0]], [0, -20*v1[1,0]], 'k')
    _,  =ax.plot([0, -20*v2[0,0]], [0, -20*v2[1,0]], 'k--')
    ax.set(xlim=[-1,1], ylim=[-1,1], xlabel="$x_1$", ylabel="$x_2$")
    ax.legend(handles=[l1, l2], loc=(1.05, .8))

def drawContour(ax, cntr, c=[0.1,0.7,0.8], ls='--'):## Plot contour
    l, = ax.plot(cntr.real, cntr.imag, ls, color=c)

    ## Plot clockwise arrows
    arrowIdx = np.linspace(start=0, stop=len(cntr), num=6, endpoint=False, dtype=int)

    for idx in range(len(arrowIdx)):
        ax.annotate("", xytext=(np.real(cntr[arrowIdx[idx]]), np.imag(cntr[arrowIdx[idx]])), 
                    xy=(np.real(cntr[arrowIdx[idx]+1]), np.imag(cntr[arrowIdx[idx]+1])),
                    arrowprops=dict(arrowstyle="->"))
    ax.set(aspect = 'equal', xlabel="$\mathfrak{Re}\{G(s)\}$", ylabel="$\mathfrak{Im}\{G(s)\}$")
    return l

def unwrap_angle(ang):
    for idx in range(ang.size - 1):
        if ang[idx+1] - ang[idx] > 180:
            ang[idx+1:] -= 360
        elif ang[idx+1] - ang[idx] < -180:
            ang[idx+1:] += 360
    return ang

import matplotlib.pyplot as plt
import numpy.random as rng
import scipy as sp
import control as cm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from IPython.display import display, Markdown

class loopShaper():
    def __init__(self, inputNoise=False, outputNoise=False):
        self.Czeros = []
        self.Cpoles = []
        self.Cgain = 1.
        self.Pzeros = [-1., self.gen_pair(1e-3, 1.5)]
        self.Ppoles = [-1e2, self.gen_pair(1e-1, .2), self.gen_pair(5e2, 3)]
        self.Pgain = 1e6
        self.OM = np.logspace(-4, 4, 500)
    
    def gen_pair(self, om0, zeta):
        q1 = -zeta*om0
        q2 = om0 * np.emath.sqrt(zeta**2 - 1)
        return q1 + q2, q1 - q2
    
    def get_response(self, Z, P, K):
        response = np.ones_like(self.S) * K
        for z in Z:
            if z == []: continue
            if type(z) == tuple: # zero pair
                response *= (self.S - z[0]) * (self.S - z[1])
            else:
                response *= (self.S - z)
        for p in P:
            if p == []: continue
            if type(p) == tuple: # pole pair
                response /= (self.S - p[0]) 
                response /= (self.S - p[1])
            else:
                response /= (self.S - p)
        return response

    def get_CLsteadygain(self):
        Lzeros, Lpoles = self.Pzeros + self.Czeros, self.Ppoles + self.Cpoles
        Lgain = self.Pgain * self.Cgain

        NL, DL, = 1, 1
        for z in Lzeros:
            if z == []: continue
            if type(z) == tuple: # zero pair
                NL *= ( - z[0]) * ( - z[1])
            else:
                NL *= ( - z)
        for p in Lpoles:
            if p == []: continue
            if type(p) == tuple: # pole pair
                DL *= ( - p[0]) * ( - p[1])
            else:
                DL *= ( - p)

        return np.abs(Lgain * NL / (DL + Lgain*NL))
    
    def unpack(self, Q):
        out = []
        for q in Q:
            if q == []: continue
            if type(q) == tuple: # pair
                out.append(q[0])
                out.append(q[1])
            else:
                out.append(q)
        return out

    
    def plot_LS(self, ax):
        [a.set_title(b) for a, b in zip([ax[3], ax[0], ax[4]], ["Nyquist plot", "Bode plots", "Step Response"])]
        [a.set_ylabel(b) for a, b in zip([ax[0], ax[1], ax[2], ax[4]],
                                         ["$|L(s)|$", r"$\angle L(s)$", "$|S(s)|$" , "$y(t)$" ])]
        ax[1].yaxis.set_major_locator(MultipleLocator(90))
        ax[2].set_xlabel(r"$\omega$")
        ax[4].set_xlabel(r"$t$")
        ax[3].set(aspect='equal',
                    xlabel=r"$\mathfrak{Re}\{L(\Gamma_s)\}$", ylabel=r"$\mathfrak{Im}\{L(\Gamma_s)\}$",
                    xlim=[-2,1], ylim=[-2, 2])
        
        self.S = self.OM*1j

        Presponse = self.get_response(self.Pzeros, self.Ppoles, self.Pgain)
        Cresponse = self.get_response(self.Czeros, self.Cpoles, self.Cgain)
        Lresponse = Cresponse * Presponse

        # Plot loop Bode
        lpM, = ax[0].loglog(self.OM, np.abs(Presponse), 'k--', alpha=.4, label="Plant")
        ax[1].semilogx(self.OM, np.angle(Presponse, deg=True), 'k', alpha=.4, label="Plant")

        lcM, = ax[0].loglog(self.OM, np.abs(Cresponse), 'k:', alpha=.4, label="Controller")
        ax[1].semilogx(self.OM, np.angle(Cresponse, deg=True), 'k:', alpha=.4, label="Controller")

        llM, = ax[0].loglog(self.OM, np.abs(Lresponse), 'k', alpha=1., label="Loop")
        ax[1].semilogx(self.OM, np.angle(Lresponse, deg=True), 'k', alpha=1., label="Loop")
        
        ax[1].legend(handles=[llM, lpM, lcM])
        [a.autoscale(enable=True, axis='x', tight=True) for a in ax[:3]]


        # Plot sensitivity
        ax[2].loglog(self.OM, np.abs(1. / (1. + Lresponse)), 'k')

        [a.axhline(1, color='r', lw=.7) for a in [ax[0], ax[2]]]

        # Plots Nyquist
        ax[3].plot(Lresponse.real, Lresponse.imag, 'k')
        ax[3].plot(Lresponse.real, -Lresponse.imag, 'k--')

        UnitCirc = np.exp(np.linspace(0,2*np.pi,100)*1j)
        ax[3].plot(UnitCirc.real, UnitCirc.imag, 'k:')
        ax[3].plot([-1], [0], 'rx')

        # Plot step response
        P = cm.zpk(self.unpack(self.Pzeros), self.unpack(self.Ppoles), self.Pgain)
        C = cm.zpk(self.unpack(self.Czeros), self.unpack(self.Cpoles), self.Cgain)
        L = cm.series(C, P)
        T = cm.feedback(L)
        t, yout = cm.step_response(L)
        ax[4].plot(t, yout, 'k')
        ax[4].autoscale(enable=True, axis='x', tight=True)
        
        # Plot steady response
        Tss = self.get_CLsteadygain()
        ax[4].axhline(Tss, color='k', ls=':')

        print(f"Steady gain is {Tss:.3E}")
        display(Markdown(f"$M_S = {np.abs(1. / (1. + Lresponse)).max():.3f}$"))

