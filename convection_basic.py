# Step 1: 1-D Linear Convection
# https://nbviewer.org/github/barbagroup/CFDPython/blob/master/lessons/01_Step_1.ipynb


# Remember: comments in python are denoted by the pound sign
import numpy                       #here we load numpy
from matplotlib import pyplot      #here we load matplotlib
import time, sys                   #and load some utilities

#from surface import draw_surf


#nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
#dx = 2 / (nx-1)
nt = 25    #nt is the number of timesteps we want to calculate
dt = .025  #dt is the amount of time each timestep covers (delta t)
c = 1      #assume wavespeed of c = 1

# u = numpy.ones(nx)  #numpy function ones()
# u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s
def linear_convection_solve(c,Lx, nx, Lt, nt):
    nx = 41  # try changing this number from 41 to 81 and Run All ... what happens?
    dx = 2 / (nx - 1)
    nt = 5  # nt is the number of timesteps we want to calculate
    dt = .025  # dt is the amount of time each timestep covers (delta t)
    c = 1  # assume wavespeed of c = 1
    Lt = dt * nt
    u = numpy.zeros(nx)  # numpy function ones()
    # u[int(.5 / dx):int(1 / dx + 1)] = 2  #setting u = 2 between 0.5 and 1 as per our I.C.s
    x = numpy.linspace(0, 1, int(nx / 2))
    u[:int(nx / 2)] = numpy.sin(numpy.pi * x)

    from surface import draw_surf

    nx = u.shape[0]
    u2D = numpy.zeros((nt,nx))
    pyplot.plot(numpy.linspace(0, 2, nx), u);

    un = numpy.ones(nx)  # initialize a temporary array

    for n in range(nt):  # loop for values of n from 0 to nt, so it will run nt times
        un = u.copy()  ##copy the existing values of u into un
        for i in range(1, nx):  ## you can try commenting this line and...
            # for i in range(nx): ## ... uncommenting this line and see what happens!
            u[i] = un[i] - c * dt / dx * (un[i] - un[i - 1])
            u2D[n,i] = u[i]
        qq = 0

    pyplot.plot(numpy.linspace(0, 2, nx), u);
    pyplot.show()
    draw_surf(Lx, nx, Lt, nt, u2D)
    return u,u2D


def convert_1D_to_2D(u,dt,nt,dx):
    nx = u.shape[0]
    u2D = numpy.zeros(nx,nt)


