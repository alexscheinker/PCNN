#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Alexander Scheinker
"""

# Import some libraries

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Input, Conv3D
from tensorflow.keras.models import Model
# from tensorflow.keras import layers

DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)


# Set paths to data and models
PATH_TO_VOLUME_DATA  = '.../Volume_Data/'
PATH_TO_MODELS       = '.../Trained_Models/'


#%%

# The 3D volumes have dimensions n_pixels*n_pixels*n_pixels
n_pixels = 128

# Import data (Small data set with first 2 time steps.)

# Charge density
Q = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Non-zero charge density locations
Qnz = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Electric field components
Ex = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Ey = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Ez = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Magnetic field components
Bx = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
By = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Bz = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Current density components
Jx = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Jy = np.zeros([2,n_pixels,n_pixels,n_pixels,1])
Jz = np.zeros([2,n_pixels,n_pixels,n_pixels,1])

# Load the data
for n_load in np.arange(2):

    Q[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Q_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    
    Qnz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Qnz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    
    Ex[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ex_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Ey[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ey_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Ez[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Ez_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    
    Bx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    By[n_load] = np.load(PATH_TO_VOLUME_DATA + f'By_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Bz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Bz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    
    Jx[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jx_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Jy[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jy_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    Jz[n_load] = np.load(PATH_TO_VOLUME_DATA + f'Jz_3D_vol_{n_pixels}_{n_load}.npy').reshape([1,n_pixels,n_pixels,n_pixels,1])
    

# Used to un-normalize PINN and no-physics CNN outputs
Bxyz_all_max = np.load(PATH_TO_VOLUME_DATA + 'Bxyz_max.npy')

# Normalize CNN inputs
J_max_max_all_128 = np.load(PATH_TO_VOLUME_DATA+'J_max_max_all_128.npy')

Jx = Jx/J_max_max_all_128
Jy = Jy/J_max_max_all_128
Jz = Jz/J_max_max_all_128

# Make 3D field data for the PINN and No-Physics CNNs for current density
Jxyz = np.zeros([2,n_pixels,n_pixels,n_pixels,3])
Jxyz[:,:,:,:,0] = Jx[:,:,:,:,0]
Jxyz[:,:,:,:,1] = Jy[:,:,:,:,0]
Jxyz[:,:,:,:,2] = Jz[:,:,:,:,0]

# Non-zero charge density locations
Qnz_3D = np.zeros([2,n_pixels,n_pixels,n_pixels,3]).astype(np.float32)
Qnz_3D[:,:,:,:,0] = Qnz[:,:,:,:,0]
Qnz_3D[:,:,:,:,1] = Qnz[:,:,:,:,0]
Qnz_3D[:,:,:,:,2] = Qnz[:,:,:,:,0]


#%%

# Define the physical space

# Some physics constants
u0 = tf.constant(4.0*np.pi*1e-7, dtype=DTYPE)
e0 = tf.constant(8.85*1e-12, dtype=DTYPE)
cc = tf.constant(2.99792e8, dtype=DTYPE)

# Physical size of the volume around the beam
x_max_all = 0.0012
x_min_all = -1*x_max_all

y_max_all = 0.0012
y_min_all = -1*y_max_all

z_max_all = 0.0022
z_min_all = -1*z_max_all

# More physics constants
me = 9.109384e-31
ce = 2.99792458e8
qe = 1.602e-19

# Size of one pixel
dx = (x_max_all-x_min_all)/(n_pixels-1)
dy = (y_max_all-y_min_all)/(n_pixels-1)
dz = (z_max_all-z_min_all)/(n_pixels-1)

# Axis for plotting
x_axis = np.linspace(x_min_all,x_max_all,n_pixels)
y_axis = np.linspace(y_min_all,y_max_all,n_pixels)
z_axis = np.linspace(z_min_all,z_max_all,n_pixels)

# Time step between saved beam volumes
dt = 1.2e-11

# Defined filters for derivatives as a convolutional layer
d_dx = np.zeros([3,3,3])
d_dx[0,1,1] = -1
d_dx[2,1,1] = 1
d_dx = tf.keras.initializers.Constant(d_dx/2)

d_dy = np.zeros([3,3,3])
d_dy[1,0,1] = -1
d_dy[1,2,1] = 1
d_dy = tf.keras.initializers.Constant(d_dy/2)

d_dz = np.zeros([3,3,3])
d_dz[1,1,0] = -1
d_dz[1,1,2] = 1
d_dz = tf.keras.initializers.Constant(d_dz/2)

# Single layerr 3D CNNs for taking partial x, y, and z derivatives 
def NN_ddx():
    X = Input(shape = (n_pixels,n_pixels,n_pixels,1))
    X_x = Conv3D(1, kernel_size=3, kernel_initializer=d_dx, strides=[1,1,1], padding='SAME', trainable=False)(X)
    d_dx_model = Model(inputs=[X], outputs=[X_x])
    return d_dx_model

def NN_ddy():
    X = Input(shape = (n_pixels,n_pixels,n_pixels,1))
    X_y = Conv3D(1, kernel_size=3, kernel_initializer=d_dy, strides=[1,1,1], padding='SAME', trainable=False)(X)
    d_dy_model = Model(inputs=[X], outputs=[X_y])
    return d_dy_model

def NN_ddz():
    X = Input(shape = (n_pixels,n_pixels,n_pixels,1))
    X_z = Conv3D(1, kernel_size=3, kernel_initializer=d_dz, strides=[1,1,1], padding='SAME', trainable=False)(X)
    d_dz_model = Model(inputs=[X], outputs=[X_z])
    return d_dz_model

# Partial Derivative CNNs
mNN_ddx = NN_ddx()
mNN_ddy = NN_ddy()
mNN_ddz = NN_ddz()

# Latent space inputs, un-used, all zeros
z_input = np.zeros([2,8,8,8,1]).astype(np.float32)

#%%

# Function that calculates B field as curl of A

def B_fields(A_model,Jx_in,Jy_in,Jz_in,z_in):
    
    # Calculate vector potential fields
    Ax1, Ax1_yL = A_model([Jx_in,z_in])
    Ay1, Ay1_yL = A_model([Jy_in,z_in])
    Az1, Az1_yL = A_model([Jz_in,z_in])
    
    # Take derivatives
    Ax1_y = mNN_ddy(Ax1)/dy
    Ax1_z = mNN_ddz(Ax1)/dz
    
    Ay1_x = mNN_ddx(Ay1)/dx
    Ay1_z = mNN_ddz(Ay1)/dz
    
    Az1_x = mNN_ddx(Az1)/dx
    Az1_y = mNN_ddy(Az1)/dy
    
    # Magnetic Fields
    Bx1 = Az1_y - Ay1_z
    By1 = Ax1_z - Az1_x
    Bz1 = Ay1_x - Ax1_y
    
    return Ax1, Ay1, Az1, Bx1, By1, Bz1

#%%


# Import trained models

A_model_name = 'PCNN_A.h5'
model_A = tf.keras.models.load_model(PATH_TO_MODELS+A_model_name)

A_Lorentz_model_name = 'PCNN_A_Lorenz.h5'
model_AL = tf.keras.models.load_model(PATH_TO_MODELS+A_Lorentz_model_name)

B_model_NoPhysics_name = 'NoPhysics_B.h5'
B_CNN_model_NoPhysics = tf.keras.models.load_model(PATH_TO_MODELS+B_model_NoPhysics_name)

B_PINN_model_name = 'PINN_B.h5'
B_PINN_model = tf.keras.models.load_model(PATH_TO_MODELS+B_PINN_model_name)


#%%

# Predict B fields using the models


# A-based model
Ax1, Ay1, Az1, Bx1, By1, Bz1 = B_fields(model_A,Jx,Jy,Jz,z_input)

# A-based Lorentz model
Ax1L, Ay1L, Az1L, Bx1L, By1L, Bz1L = B_fields(model_AL,Jx,Jy,Jz,z_input)

# No physics model
Bxyz_now, yL_temp = B_CNN_model_NoPhysics.predict([Jxyz, z_input, Qnz_3D])

# PINN model
Bxyz_now_PINN, yL_temp_PINN = B_PINN_model.predict([Jxyz, z_input, Qnz_3D])


#%%

# Cut off edges of low-density 

Bx1 = Bx1*Qnz
By1 = By1*Qnz
Bz1 = Bz1*Qnz

Bx1L = Bx1L*Qnz
By1L = By1L*Qnz
Bz1L = Bz1L*Qnz

# These two predictions also have to be un-normalized for comparrison
Bxyz_now = Bxyz_now*Qnz_3D*Bxyz_all_max

Bxyz_now_PINN = Bxyz_now_PINN*Qnz_3D*Bxyz_all_max

#%%

# Location of highest charge density
B_max_proj = np.max(np.max(np.abs(Bx[0,:,:,:,0]),0),0)
plt.plot(B_max_proj,'.')

#%%

# Calculate B-field divergence values 
# divB = dB/dx + dB/dy + dB/dz

# Calculate the divergence of the magnetic field from the PCNN A model
BxA_x = mNN_ddx(Bx1[0:1])
ByA_y = mNN_ddy(By1[0:1])
BzA_z = mNN_ddz(Bz1[0:1])
divBA = BxA_x/dx + ByA_y/dy + BzA_z/dz

# Calculate the divergence of the magnetic field from the PCNN A Lorenz model
BxAL_x = mNN_ddx(Bx1L[0:1])
ByAL_y = mNN_ddy(By1L[0:1])
BzAL_z = mNN_ddz(Bz1L[0:1])
divBAL = BxAL_x/dx + ByAL_y/dy + BzAL_z/dz

# Calculate the divergence of the magnetic field from the no physics B model
BxNN_x = mNN_ddx(Bxyz_now[0:1,:,:,:,0])
ByNN_y = mNN_ddy(Bxyz_now[0:1,:,:,:,1])
BzNN_z = mNN_ddz(Bxyz_now[0:1,:,:,:,2])
divBNN = BxNN_x/dx + ByNN_y/dy + BzNN_z/dz

# Calculate the divergence of the magnetic field from the PINN B model
BxPINN_x = mNN_ddx(Bxyz_now_PINN[0:1,:,:,:,0])
ByPINN_y = mNN_ddy(Bxyz_now_PINN[0:1,:,:,:,1])
BzPINN_z = mNN_ddz(Bxyz_now_PINN[0:1,:,:,:,2])
divBPINN = BxPINN_x/dx + ByPINN_y/dy + BzPINN_z/dz


#%%

# Image of the charge density slice

plt.figure(1, figsize=(5,4))
plt.imshow(Q[0,:,:,40,0]/np.max(Q[0,:,:,40,0]),aspect='auto',extent=(-x_max_all/1e-3,x_max_all/1e-3,-y_max_all/1e-3,y_max_all/1e-3))
plt.xlabel('$\Delta x [mm]$')
plt.ylabel('$\Delta y [mm]$')
plt.colorbar()
plt.tight_layout()

#%%

# Plot 2D slices of all field divergences

vminmax = 20

plt.figure(2, figsize=(15,15))

plt.subplot(4,3,1)
plt.imshow(divBA[0,:,:,40,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, PCNN')
plt.colorbar()

plt.subplot(4,3,2)
plt.imshow(divBA[0,:,64,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, PCNN')
plt.colorbar()

plt.subplot(4,3,3)
plt.imshow(divBA[0,64,:,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, PCNN')
plt.colorbar()


plt.subplot(4,3,4)
plt.imshow(divBAL[0,:,:,40,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, PCNN Lorenz')
plt.colorbar()

plt.subplot(4,3,5)
plt.imshow(divBAL[0,:,64,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, PCNN Lorenz')
plt.colorbar()

plt.subplot(4,3,6)
plt.imshow(divBAL[0,64,:,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, PCNN Lorenz')
plt.colorbar()



plt.subplot(4,3,7)
plt.imshow(divBNN[0,:,:,40,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, No Physics')
plt.colorbar()

plt.subplot(4,3,8)
plt.imshow(divBNN[0,:,64,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, No Physics')
plt.colorbar()

plt.subplot(4,3,9)
plt.imshow(divBNN[0,64,:,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, No Physics')
plt.colorbar()



plt.subplot(4,3,10)
plt.imshow(divBPINN[0,:,:,40,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, PINN')
plt.colorbar()

plt.subplot(4,3,11)
plt.imshow(divBPINN[0,:,64,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, PINN')
plt.colorbar()

plt.subplot(4,3,12)
plt.imshow(divBPINN[0,64,:,:,0], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, PINN')
plt.colorbar()


plt.tight_layout()

#%%

# Cut off beam by thresholding to get rid of the bad edges

iii = np.where(Q[0]>1e-15)
ZZZ = np.zeros([1,128,128,128])
ZZZ[0,iii[0],iii[1],iii[2]] = 1.0


#%%

# Plot 2D slices of all field divergences

vminmax = 20

plt.figure(2, figsize=(15,15))

plt.subplot(4,3,1)
plt.imshow(divBA[0,:,:,40,0]*ZZZ[0,:,:,40], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, PCNN')
plt.colorbar()

plt.subplot(4,3,2)
plt.imshow(divBA[0,:,64,:,0]*ZZZ[0,:,64,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, PCNN')
plt.colorbar()

plt.subplot(4,3,3)
plt.imshow(divBA[0,64,:,:,0]*ZZZ[0,64,:,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, PCNN')
plt.colorbar()


plt.subplot(4,3,4)
plt.imshow(divBAL[0,:,:,40,0]*ZZZ[0,:,:,40], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, PCNN Lorenz')
plt.colorbar()

plt.subplot(4,3,5)
plt.imshow(divBAL[0,:,64,:,0]*ZZZ[0,:,64,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, PCNN Lorenz')
plt.colorbar()

plt.subplot(4,3,6)
plt.imshow(divBAL[0,64,:,:,0]*ZZZ[0,64,:,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, PCNN Lorenz')
plt.colorbar()


plt.subplot(4,3,7)
plt.imshow(divBNN[0,:,:,40,0]*ZZZ[0,:,:,40], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, No Physics')
plt.colorbar()

plt.subplot(4,3,8)
plt.imshow(divBNN[0,:,64,:,0]*ZZZ[0,:,64,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, No Physics')
plt.colorbar()

plt.subplot(4,3,9)
plt.imshow(divBNN[0,64,:,:,0]*ZZZ[0,64,:,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, No Physics')
plt.colorbar()


plt.subplot(4,3,10)
plt.imshow(divBPINN[0,:,:,40,0]*ZZZ[0,:,:,40], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,y)$, PINN')
plt.colorbar()

plt.subplot(4,3,11)
plt.imshow(divBPINN[0,:,64,:,0]*ZZZ[0,:,64,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(x,z)$, PINN')
plt.colorbar()

plt.subplot(4,3,12)
plt.imshow(divBPINN[0,64,:,:,0]*ZZZ[0,64,:,:], aspect='auto', vmin=-vminmax, vmax=vminmax)
plt.title(r'$\nabla\cdot B(y,z)$, PINN')
plt.colorbar()

plt.tight_layout()



















