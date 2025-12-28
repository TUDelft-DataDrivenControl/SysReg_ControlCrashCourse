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
    ax.plot(cntr.real, cntr.imag, ls, color=c)
    # ax.set(aspect='equal', ylabel="Im", xlabel="Re")

    ## Plot clockwise arrows
    arrowIdx = np.linspace(start=0, stop=len(cntr), num=6, endpoint=False, dtype=int)

    for idx in range(len(arrowIdx)):
        ax.annotate("", xytext=(np.real(cntr[arrowIdx[idx]]), np.imag(cntr[arrowIdx[idx]])), 
                    xy=(np.real(cntr[arrowIdx[idx]+1]), np.imag(cntr[arrowIdx[idx]+1])),
                    arrowprops=dict(arrowstyle="->"))
    ax.set_aspect('equal')

def unwrap_angle(ang):
    for idx in range(ang.size - 1):
        if ang[idx+1] - ang[idx] > 180:
            ang[idx+1:] -= 360
        elif ang[idx+1] - ang[idx] < -180:
            ang[idx+1:] += 360
    return ang