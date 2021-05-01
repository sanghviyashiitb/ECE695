# ECE695
ECE695 Generative Modelling Course Project

To run the scripts first download the neural recording data through the following terminal command:

 ```console
wget https://zenodo.org/record/3854034/files/indy_20160411_01.mat
```

There are two "main_.." files in the repository. 
1. *main_kf_pf.m* : Compares the performance of Kalman filter and Particle Filter
2. *main_poiss_bernoulli.m*: Compares the performance for different emission proabilities: Gaussian, Poisson, and Binomial using a particle filter
