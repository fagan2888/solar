# brute force N-body simulation

import sys
import numpy as np
import pandas as pd
import vector_tools as vt
from scipy.constants import G as G0
import matplotlib.pyplot as plt
plt.ion()
new = np.newaxis

# algo params
rep = 100
box = 500 # (AU)
size = lambda m: np.r_[50, 1e5*m[1:]**(1/1.5)]

# utils
def halve(x, s):
    n0 = len(x) >> 1
    n1 = np.sum(s)
    xp = np.zeros(n0)
    xp[:n1] = x[s]
    return xp

# astro params
N = 2**8 # number of particles
M = 1 # sol masses
dt = 0.01 # years
crad = 1e-1 # AU

# scale
L = 1.496e11 # length scale (1 AU)
K = 1.989e30 # mass scale (1 sol mass)
T = 3.154e8 # time step (1 year)

# units
G = G0*(1/L**3)*(K)*(T**2)

# state
st = vt.Bundle()

# initialize
st.mas = (1e24/K)*np.exp(2*np.random.rand(N)-1)
st.posx = (1e13/L)*np.random.randn(N)
st.posy = (1e13/L)*np.random.randn(N)
rad = np.sqrt(st.posx**2+st.posy**2)
spd = np.sqrt(G*M/rad)
st.velx = 0.7*(st.posy/rad)*spd
st.vely = -0.7*(st.posx/rad)*spd

# perturb
st.velx += 1*(2*np.random.rand(N)-1)
st.vely += 1*(2*np.random.rand(N)-1)

# the star
st.mas[0] = M
st.posx[0] = 0
st.posy[0] = 0
st.velx[0] = 0
st.vely[0] = 0

# plot
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(st.posx, st.posy, s=size(st.mas))
ax.set_xlim(-box, box)
ax.set_ylim(-box, box)

k = 0
while True:
    if k % 100 == 0:
        print('rep %d' % k)

    # reduce
    sel = st.mas > 0
    n = np.sum(sel)
    if n <= (N >> 1):
        print('halving, n = %d' % (N>>1))
        N = N >> 1
        st.mas = halve(st.mas, sel)
        st.posx = halve(st.posx, sel)
        st.posy = halve(st.posy, sel)
        st.velx = halve(st.velx, sel)
        st.vely = halve(st.vely, sel)
        ax.cla()
        sc = ax.scatter(st.posx, st.posy, s=size(st.mas))
        ax.set_xlim(-box, box)
        ax.set_ylim(-box, box)

    # distances
    delx = st.posx[new, :] - st.posx[:, new]
    dely = st.posy[new, :] - st.posy[:, new]
    rmat = np.sqrt(delx**2+dely**2)

    # collisions
    coli, colj = (rmat<crad).nonzero()
    ncol = len(coli)
    if ncol > 0:
        sel = (coli<colj) & (st.mas[coli]>0) & (st.mas[colj]>0)
        ncol = np.sum(sel)
        if ncol > 0:
            print(np.sum(st.mas>0))
            coli = coli[sel]
            colj = colj[sel]
            tmas = st.mas[coli] + st.mas[colj]
            st.posx[coli] = (st.mas[coli]*st.posx[coli]+st.mas[colj]*st.posx[colj])/tmas
            st.posy[coli] = (st.mas[coli]*st.posy[coli]+st.mas[colj]*st.posy[colj])/tmas
            st.velx[coli] = (st.mas[coli]*st.velx[coli]+st.mas[colj]*st.velx[colj])/tmas
            st.vely[coli] = (st.mas[coli]*st.vely[coli]+st.mas[colj]*st.vely[colj])/tmas
            st.mas[coli] = tmas
            st.mas[colj] = 0

    # calculate accel
    fmat = G*st.mas/rmat**2
    accx = (delx/rmat)*fmat
    accy = (dely/rmat)*fmat
    accx[np.isnan(accx)] = 0
    accy[np.isnan(accy)] = 0
    movx = np.sum(accx, 1)
    movy = np.sum(accy, 1)

    # immobile star
    movx[:] -= movx[0]
    movy[:] -= movy[0]

    # implement accel
    st.posx += dt*st.velx
    st.posy += dt*st.vely
    st.velx += dt*movx
    st.vely += dt*movy

    # update plot
    if k % rep == 0:
        sc.set_offsets(np.dstack([st.posx, st.posy]))
        sc.set_sizes(size(st.mas))
        fig.canvas.draw()

    k += 1

