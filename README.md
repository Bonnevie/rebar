# rebar
A Python implementation of the gradient REBAR estimator.

This Python module implements the REBAR gradient estimator by way of the function REBAR(), which takes a function and appropriate parameters as input and then uses Tensorflow to compute the appropriate gradients.
The auxiliary functions necessary for computing REBAR estimators for both binary and categorical random variables are already implemented.
Finally, the module supports dynamic REBAR: taking gradients with respect to estimator variance so that the method can be tuned automatically.
