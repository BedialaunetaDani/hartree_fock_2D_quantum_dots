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
6. @dbedialaunetar implemented the algorithm for solving the Roothaan equations [commit](65044a09cd30bb0807ed1a28305c181ba4336bbd)

--------

The aim of this project is to use the Hartree-Fock method to compute some properties of two-dimensional quantum dots. The theory related to this project is quite extensive so, before explaining the implemented code, we will first introduce the necessary concepts to understand its structure.

**Theoretical background**


List of indices:
- i,j: loop over electrons (N_e = number of electrons)
- k: loop over basis elements (N_b = number of basis elements)
- p,q,r,s: loop the matrices (so also counting the basis N_b X N_b)


 The hamiltonian of $`N_e`$ electrons in such a system is given by 

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

In order to compute this integrals we will use the Monte Carlo Integration implemented in the previous project. We can simply sample the integrand with a random walker and then do the corresponding sum. We will also have to implement a numerical approach for the laplacian, as we did in the previous project. 

In order to solve the iterative generalized eigenvalue equation, we start with an initial choice of coefficients $`\bm{C^{(0)}}`$, with which we compute the Fock matrix. Afterwards, we solve the generalized eigenvalue equation

```math
\bm{F(C^{(k-1)})}\bm{C^{(k)}}=\bm{SC^{(k)}}\epsilon^{(k)}, \quad k=1,2,3,...,
```

iteratively. The solution is a vector of eigenvalues and a vector of coefficientes for each of the eigenvalues. There exist various stopping criteria but, esentially, they consist on looking at whether the changes to the coefficientes or energies from one iteration to another are small enough. Although it is subject to change, we will use

```math
\text{max}|(\epsilon^{(k-1)}-\epsilon^{(k)})/\epsilon^{(k)}|< \text{precision},
```

where ''max'' refers to the maximum value of the vector. The generalized eigenvalue problem is solved using the scipy function eigh(F,S).

Following Jos' book, the groundstate energy is then computed through

```math
E = \frac{1}{2}\left[ \sum_{rs}h_{rs} P_{rs} + \sum_k \epsilon_k \right]
```

where P_rs stands for a density matrix in the RHF from

```math
P_{pq} = 2 \sum_k C_{pk} C_{qk}^*
```

----------

**Implemented code**

The first step towards building a Hartree-Fock solver is introducing the base functions, which is done in [basis_set.py](63260f150c559d84a7e8847b0c4bac91b906c433).

Once this is done we are prepared to compute the necessary integrals $`h_{pq}`$ and $`\braket{pr|g|qs}`$, for which we import the [Monte Carlo library](55eba781cacc37ae9e75f58e05325489f89b0982) implemented in the previous project.

This leads to the computation of several matrices, which form a generalized eigenvalue problem that is solved iteratively using eigh(F,S). This was implemented in the function [solve_roothaan_equations](65044a09cd30bb0807ed1a28305c181ba4336bbd).

The basis functions are built from Hermite polynomials that can be computed in several ways: (1) using the already existing polynomials from `scipy` or `numpy`, or (2) build the polynomials using a recursive method. We then implemented [timing.py](558046059d7503cc8c8dfd101ced83fab0027a61), where we analyse which of those is less time consuming. The result is simply that it is better to use the build in `scipy` functions specially when working with high order Hermite polynomials. This file is also where we will analyse the efficiency of the code from now on. 

Finally, in [checks.ipynb](558046059d7503cc8c8dfd101ced83fab0027a61), we will perform different checks to ensure that already implemented code works as expected. This week it contains a check of the basis functions in which we plot several Hermite polynomials and perform a basic integrations in order to ensure that they are properly normalized, the outcome was in agreement with our expectations. You can refer to the jupyter notebook to see the extended results.

## Week 2

### Bullet List

1. Implement functions to calculate the $`<pr|g|qs>`$ and $`h_{pq}`$ integrals using Monte Carlo (from the previous project) (@abermejillo)
1. Implement the necessary code that, due to symmetries, avoids redundant and null integrals, and stores them in a file correctly indexed. (@dbedialaunetar)
1. Create a function to get the J and K matrices and then the Fock matrix for a given set of coefficients (@mserraperalta)
1. Implement the many-body ground state energy function and the density matrix (@mserraperalta)
1. Select a set of initial parameters and check that the Hartree-Fock code does not have any bugs (@abermejillo, @mserraperalta, @dbedialaunetar)
1. Find some test to verify that the Hartree-Fock code works correctly (helium atom, for example) (@abermejillo, @mserraperalta, @dbedialaunetar)


### Progress

1. @abermejillo made some adaptations to the code from last project but found some issues: more on this bellow.
2. @dbedialaunetar implmented the necessary code that stores values of the integrals avoiding reduncancies: [commit](21c180a03c2ce204f1ad1f65691bcb974c7bb8a5)
3. @mserraperalta implemented code that takes the values of the integrals from a file and generates the matrices J, K and F: [commit](f3fae22649ce757932faaadebb16e41399ed2b65)
4. @mserraperalta implemented functions that compute the groundstate energy and the density matrix: [commit](f3fae22649ce757932faaadebb16e41399ed2b65)
5. Not done because the integrals still need to be solved
6. @mserraperalta and @abermejillo started implementing code to test in the Helium atom: [commit1](7d6990f9d0121e4560ccd40b72e5abee5d0c9112) and [commit2](d0e2456285a6764d02e88bea9d8f1357f6ab9779)

The most complicated step of the HF implementation is the computation of the two electron integrals. They correspond to `$\braket{pr|g|qs}$`, defined in the precious week. Our intention was to adapt the code from the previous project to implement Monte Carlo integration. however, there's a step that needs to be done, which is convert the integrand into a product of a probability distribution times another function. A possible way of doing it would be to express the integrand as

```math
\braket{pr|g|qs} = \int d\bm{r_1} d\bm{r_2} \phi_p(\bm{r_1})\phi_r(\bm{r_2})r_{12}^{-1}  \phi_q(\bm{r_1})\phi_s(\bm{r_2}) 
=  \int d\bm{r_1} d\bm{r_2} |\phi_p(\bm{r_1})\phi_r(\bm{r_2})|^2  \frac{\phi_q(\bm{r_1})\phi_s(\bm{r_2})}{\phi_p(\bm{r_1})\phi_r(\bm{r_2})}r_{12}^{-1}.
```

However we are not sure whether this procedure is correct. Another way would be to follow [this paper](https://aip.scitation.org/doi/10.1063/1.5114703). In it they simply introduce another probability distribution based on a simple gaussian. Nevertheless, the paper continues doing other implementations to actually get its results, so by only implementing the first part we are not assured we will get reasonable results. 

We will have to decide how to solve this issue soon and implement a way to compute this integrals.

On the other hand the rest of the Hartree Fock solver is already implemented. We have a class that aranges the integrals and gets them into a file from which they can be read to compute the Fock matrix. From there, the Self Consistent iteration can be performed and the energy and density matrix computed. 

Finally, the pieces of code that do work were used to INTRODUCE COMPUTATIONS OF HeH with integrals from the webpage!!!!!!


(due 30 May 2022, 23:59)

## Week 3

### Bullet List


### Progress


(due 6 June 2022, 23:59)

