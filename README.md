# PCNN
Physics Constrained Neural Networks

This work presents a physics-constrained neural network (PCNN) approach to solving Maxwell’s equations for the electromagnetic fields of intense relativistic charged particle beams. 

We create a 3D convolutional PCNN to map time-varying current and charge densities J(r,t) and ρ(r,t) to vector and scalar potentials A(r,t) and φ(r,t) from which we generate electromagnetic fields according to Maxwell’s equations: 
B = ∇×A, 
E = −∇φ −∂A/∂t. 

Our PCNNs satisfy hard constraints, such as ∇ · B = 0, by construction. Soft constraints push A and φ towards satisfying the Lorenz gauge.

Included are the code to define and train the 3D convolutional neural networks as well as data sets for testing of the algorithm.

The PCNN setup for hard constrained E and B field generation with a soft Lorenz gauge constraint is shown in the Figure below.
![plot](https://github.com/alexscheinker/PCNN/blob/main/3D_PCNN_Lorenz.png)

One example of generated E and B fields for a 3D electron bunch of varying charge and current density is shown below.
![plot](https://github.com/alexscheinker/PCNN/blob/main/3D_EB_Fields.png)

<p align="center">
  <img width="600" height="800" src="https://github.com/alexscheinker/PCNN/blob/main/3D_EB_Fields.png">
</p>
