# PCNN
Physics Constrained Neural Networks

This work presents a physics-constrained neural network (PCNN) approach to solving Maxwell’s equations for the electromagnetic fields of intense relativistic charged particle beams. 

We create a 3D convolutional PCNN to map time-varying current and charge densities J(r,t) and ρ(r,t) to vector and scalar potentials A(r,t) and φ(r,t) from which we generate electromagnetic fields according to Maxwell’s equations: 
B = ∇×A, 
E = −∇φ −∂A/∂t. 

Our PCNNs satisfy hard constraints, such as ∇ · B = 0, by construction. Soft constraints push A and φ towards satisfying the Lorenz gauge.

Included are the code to define and train the 3D convolutional neural networks as well as data sets for testing of the algorithm.

The PCNN setup for hard constrained E and B field generation with a soft Lorenz gauge constraint is shown in the Figure below.
<p align="center">
  <img width="800" height="250" src="https://github.com/alexscheinker/PCNN/blob/main/3D_PCNN_Lorenz.png">
</p>

One example of generated E and B fields for a 3D electron bunch of varying charge and current density is shown below.
<p align="center">
  <img width="600" height="800" src="https://github.com/alexscheinker/PCNN/blob/main/3D_EB_Fields.png">
</p>

We look at an (x,y) slice in the middle of the beam, as shown with normalized charge density in the image below.
<p align="center">
  <img width="350" height="300" src="https://github.com/alexscheinker/PCNN/blob/main/XY_Beam_Slice.png">
</p>

In the attached code "Check_Divergence" we generate the magnetic field B using a regular CNN without physics constraints, a PINN-based B field, a PCNN-based B field, and the Lorenz PCNN B field. We then calculate the divergence of the B field in each case, as shown below.

<p align="center">
  <img width="800" height="800" src="https://github.com/alexscheinker/PCNN/blob/main/Divergence_Compared.png">
</p>

Note that all of the divergence values for all fields have problems at the edges of the beam distribution where the beam density is low. This is due to the fact that the code used for this work only generated (E,B) fields at particle locations and therefore at low particle density regions the fields un-physically suddenly vanish and destroy  the divergence calculation.

However, in the core of the beam where the charge density is non-zero, it is clear that the PCNN-based B fields have almost zero (within numerical limitations) divergence while both the no physics and the PINN-based fields have non-zero divergence. This is also described in the paper.

One way to see this differnce more cleanly is to use a threshold, to cut off fields at regions of sufficiently low charge density and then plot the divergence only there, as shown in the figure below.

<p align="center">
  <img width="800" height="800" src="https://github.com/alexscheinker/PCNN/blob/main/Divergence_Charge_Threshold_Compared.png">
</p>
