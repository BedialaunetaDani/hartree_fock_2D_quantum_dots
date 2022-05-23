# Weekly progress journal

## Week 1

### Bullet List

1. Create functions for the single energy wave functions @mserraperalta
2. Add spin to single energy wave functions @mserraperalta
3. Search information about Fock matrix and how to implement it @abermejillo
4. Select a method for integrating and start implementing the calculation of the elements of the Fock matrix @abermejillo
6. Search information for a method for solving the Roothaan equations and self-consistent field @dbedialaunetar
7. Select a method for solving the Roothaan equations and start implementing it @dbedialaunetar
8. Join all information and preliminary results from this week in the presentation's slides @dbedialaunetar @mserraperalta @abermejillo

### Progress

1. @mserraperalta added the single electron wave functions: [commit](33ccb28197907fb20752c70115834976e01bc022)
2. The spin does not need to be added, it is taken into account separately (@mserraperalta)
3. There's information on how to compute it in Jos' book, [this link](https://adambaskerville.github.io/posts/HartreeFockGuide/) and this [github repository](https://schoyen.github.io/tdhf-project-fys4411/task-2-ghf-solver.html) (@abermejillo)
4. The integration method will be recycled from the previous project, Monte Carlo Integration: [commit](55eba781cacc37ae9e75f58e05325489f89b0982)
5. There's information on how to solve the Roothaan equations in the same links as 3 (@dbedialaunetar)
6. The library used to diagonalize will be scipy and the convergence criteria is explained in Jos' book (@dbedialaunetar)

------------
Indices:
- i,j: loop over electrons (N_e = number of electrons)
- k: loop over basis elements (N_b = number of basis elements)
- p,q,r,s: loop the matrices (so also counting the basis N_b X N_b)
--------------

The aim of this project is to use the Hartree-Fock method to compute some properties of two-dimensional quantum dots. The hamiltonian of $`N_e`$ electrons in such a system is given by 

```math
\hat{H}(\bm{r_i})=\sum_{i=1}^{N_e}h_i(\bm{r_i}) + \sum_{i<j}^{N_e} \frac{1}{\bm{r_{ij}}}, 
```

where

```math
h_i(\bm{r_i}) = -\frac{1}{2}\nabla_i^2 + \frac{1}{2}\omega_1(x_i^2+y_i^2) + \frac{1}{2}\omega_2 z_i^2.
```

We can recognise that the electrons will be confined by harmonic potentials, strongly in the z axis to make it two dimensional and more weakly in the xy plane defining the quantum dot.

The basis with which the Hartree-Fock algorithm will be implemented is the solution to Schrödinger's equation of the single particle hamiltonians $`h(r_i)`$, which are nothing but Hermite polynomials

```math
\phi_k (x,y,z) =\psi(x,n_x) \psi (y, n_y) \psi (z, n_z),
```

where $`\psi(x,n)=\frac{1}{\sqrt{2^n n! \pi }}e^{-x^2/2}\mathcal{H}_n(x)`$ and $`\mathcal{H}_n(x)=(-1)^ne^{x^2}\frac{d^n}{dx^n}\left(e^{-x^2}\right)`$.

With this basis we need to solve the Roothan Equations 

```math
\bm{FC}=\bm{SC}\epsilon.
```

We first need to compute the matrices $`\bm{F}`$ and $`\bm{S}`$. $`\bm{F}`$ is the Fock matrix and is computed in the following way

```math
F_{pq} = h_{pq} + 2J_{pq} - K_{pq},
```

where $`h_{pq}`$ is the single electron matrix element, $`J_{pq}`$ is the coulomb element and $`K_{pq}`$ the exchange element. They are given by

```math
h_{pq} = \int d\bm{r} \phi_p^*(\bm{r}) \left[-\frac{1}{2}\nabla_k^2 + \frac{1}{2}\omega_1(x_k^2+y_k^2) + \frac{1}{2}\omega_2 z_k^2\right]\phi_q(\bm{r}) ,
```

```math
J_{pq} = \sum_k \sum_{rs} C_{rk}^*C_{sk} \braket{pr|g|qs},
```
and

```math
K_{pq} = \sum_k \sum_{rs} C_{rk}^*C_{sk} \braket{pr|g|sq},
```

where the braket is

```math
\braket{pr|g|qs} = \int d\bm{r_1} d\bm{r_2} \phi_p(\bm{r_1})\phi_r(\bm{r_2})r_{12}^{-1}  \phi_q(\bm{r_1})\phi_s(\bm{r_2}).
```

Finally, $`\bm{S}`$ is the overlap matrix given by 

```math
\bm{S}_{pq} = \int \phi_{p}^*(\bm{r}) \phi_{q}(\bm{r}) d\bm{r},
```

which takes account of the normalization. 

This integrals are the bottleneck of the program and need to be properly computed beforehand and stored in a text file. Many integrals can be avoided due to symmetry arguments, reducing considerably the amount of computations required. As a drawback the book keeping gets more difficult.

In order to compute this integrals we will use the Monte Carlo Integration implemented in the previous project. We can simply sample the integrand with a random walker and then do the corresponding sum. We will have to implement a numerical approach for the laplacian, as we did in the previous project. 

In order to solve the iterative generalized eigenvalue equation, we start with an initial choice of coefficients $`\bm{C^{(0)}}`$, with which we compute the Fock matrix. Afterwards, we solve the generalized eigenvalue equation

```math
\bm{F(C^{(k-1)})}\bm{C^{(k)}}=\bm{SC^{(k)}}\epsilon^{(k)}, \quad k=1,2,3,...,
```

iteratively. The solution is a vector of eigenvalues and a vector of coefficientes for each of the eigenvalues. There exist various stopping criteria but, esentially, they consist on looking at whether the changes to the coefficientes or energies from one iteration to another are small enough. Although it is subject to change, we will use

```math
\text{max}|(\epsilon^{(k-1)}-\epsilon^{(k)})/\epsilon^{(k)}|< \text{precision},
```

where ''max'' refers to the maximum value of the vector. The generalized eigenvalue proble is solved using the scipy function eigh(F,S).

Following Jos' book, the groundstate energy is then computed through

```math
E = \frac{1}{2}\left[ \sum_{rs}h_{rs} P_{rs} + \sum_k \epsilon_k \right]
```

where P_rs stands for a density matrix in the RHF from

```math
P_{pq} = 2 \sum_k C_{pk} C_{qk}^*
```


## Week 2

### Bullet List


### Progress


(due 30 May 2022, 23:59)

## Week 3

### Bullet List


### Progress


(due 6 June 2022, 23:59)

