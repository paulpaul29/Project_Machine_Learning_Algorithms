# Project_Machine_Learning_Algorithms

In this repository, there are three main working directories:

1. ReactionRateCoefficientsRegression
2. TransportCoefficientsRegression
3. PINN_Euler_1d_shock_STS

## Pre-requisites

* python 3 (with python 2 there could be some problems)
* scikit-learn: https://scikit-learn.org/stable/install.html
* Tensorflow 1.4 or 1.5 (superior may not work properly)
* Keras

## Reaction Rate Coefficients Regression
In this directory, we do the regression of reaction rate coefficients according to the state-to-state (STS) theory.

In `Utilities` there are few functions for plotting (not interesting for you).
In `docs` there are some documents which may be useful for you to read.

## How to run?
From `Project_Machine_Learning_Algorithms/ReactionRateCoefficientsRegression/DR/src` run:

~~~~~
./run_regression.sh
~~~~~

This will learn one vibrational level at the time.

Otherwise,

~~~~~
./run_regression_multioutput.sh
~~~~~

This will learn all the dataset at once.

In both cases, the model, scalers and figures will be saved in `model`, `scaler` and `pdf` folders, respectively.

### Note
Only the scripts for dissociation are present. You can write the scripts for recombination (DR).
In the same way, I only considered DR processes (datasets) here, neglecting VT, VV, VV2, and ZR. You can also try them.


## Transport Coefficients Regression
In this directory, we do the regression of transport coefficients according to the state-to-state (STS) theory.
In particular, we would like to predict:

* shear viscosity
* bulk viscosity
* thermal conductivity
* thermal diffusion
* mass diffusion

For a rigorous definition of the transport coefficients, please refer to the book:
Nagnibeda, E., & Kustova, E. (2009). Non-equilibrium reacting gas flows: kinetic theory of transport and relaxation processes.
Springer Science & Business Media.


## Physically Informed Neural Network (PINN)
In this directory, we do want to try to use PINNs to solve the Euler equations for a one-dimensional shock flow problem,
according to the STS formuation.

There may be useful to take a look at some other repositories to better understand PINNs, for example:

* https://github.com/maziarraissi
* https://github.com/maziarraissi/PINNs.git
* https://github.com/jhssyb/PINN--NUS.git
* https://github.com/lululxvi/deepxde.git
* https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs.git

