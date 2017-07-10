# brute force N-body simulation

import sys
import numpy as np
from scipy.constants import G as G0
import matplotlib.pyplot as plt
plt.ion()
new = np.newaxis

# algo params
box = 400 # (AU)

# astro params
N = 2**8 # number of particles
M = 1 # sol masses
dt = 0.1 # years

# scale
L = 1.496e11 # length scale (1 AU)
K = 1.989e30 # mass scale (1 sol mass)
T = 3.154e8 # time step (1 year)

# units
G = G0*(1/L**3)*(K)*(T**2)

# initialize
mas = (1e18/K)*np.exp(20*np.random.rand(N))
posx = (1e13/L)*(2*np.random.rand(N)-1)
posy = (1e13/L)*(2*np.random.rand(N)-1)
rad = np.sqrt(posx**2+posy**2)
spd = np.sqrt(G*M/rad)
velx = (posy/rad)*spd
vely = -(posx/rad)*spd

# perturb
velx += 1*(2*np.random.rand(N)-1)
vely += 1*(2*np.random.rand(N)-1)

# the star
mas[0] = M
posx[0] = 0
posy[0] = 0
velx[0] = 0
vely[0] = 0

# plot
fig, ax = plt.subplots(figsize=(5, 5))
sc = ax.scatter(posx, posy, s=np.log(K*mas)-30)
ax.set_xlim(-box, box)
ax.set_ylim(-box, box)

while True:
    # calculate accel
    delx = posx[new, :] - posx[:, new]
    dely = posy[new, :] - posy[:, new]
    rmat = np.sqrt(delx**2+dely**2)
    fmat = G*mas/rmat**2
    accx = (delx/rmat)*fmat
    accy = (dely/rmat)*fmat
    np.fill_diagonal(accx, 0)
    np.fill_diagonal(accy, 0)
    movx = np.sum(accx, 1)
    movy = np.sum(accy, 1)

    # immobile star
    movx[0] = 0
    movy[0] = 0

    # implement accel
    posx += dt*velx
    posy += dt*vely
    velx += dt*movx
    vely += dt*movy

    # update plot
    sc.set_offsets(np.dstack([posx,posy]))
    fig.canvas.draw()

