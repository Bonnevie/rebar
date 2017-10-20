# rebar
A Python implementation of the gradient REBAR estimator.

This Python module implements the REBAR gradient estimator by way of the function REBAR(), which takes a function and appropriate parameters as input and then uses Tensorflow to compute the appropriate gradients.
The auxiliary functions necessary for computing REBAR estimators for both binary and categorical random variables are already implemented.
Finally, the module supports dynamic REBAR: taking gradients with respect to estimator variance so that the method can be tuned automatically.

Running the module as a script starts a small demo illustrating the advantages of using REBAR

![Result of running the REBAR demo.](rebar_demo.png)

Here, REBAR clearly outperforms the standard score estimator in terms of variance, and optimizing the hyperparameters reduces the variance further.
