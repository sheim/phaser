# Phaser
The phaser module provides an implementation of the phase estimation algorithm of "Estimating the phase of synchronized oscillators";     S. Revzen &amp; J. M. Guckenheimer; Phys. Rev. E; 2008, v. 78, pp. 051907    doi: 10.1103/PhysRevE.78.051907 

Phaser takes in multidimensional data from multiple experiments and fits the
parameters of the phase estimator, which may then be used on new data or the
training data. The output of Phaser is a phase estimate for each time sample
in the data. This phase estimate has several desirable properties, such as:
1. d/dt Phase is approximately constant
2. Phase estimates are robust to measurement errors in any one variable
3. Phase estimates are robust to systematic changes in the measurement error

# Requirements
This version of phaser has been tested with python 3.5, numpy 1.11.1, matplotlib 1.5.1 and scipy 0.17.1, which are required modules. Currently python 2.7 is not supported.