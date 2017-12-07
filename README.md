# REBAR/RELAX
A Python implementation of the gradient REBAR and RELAX estimator.

This Python module implements the RELAX gradient estimator by way of the function RELAX(), which takes Tensorflow tensors corresponding to the scalar quantity to be differentiated along with tensors representing a control variate and the same control variate using a conditional noise variable. The function then uses Tensorflow to compute the appropriate gradients.
The REBAR estimator is a canonical instance of RELAX which uses a relaxed sampling procedure for the control variate.
The auxiliary functions necessary for computing REBAR/RELAX estimators for both binary and categorical random variables are already implemented.
Finally, the module supports dynamic REBAR/RELAX: taking gradients with respect to estimator variance so that the method can be tuned automatically.

There are two modules:
relax - which implements the RELAX estimator and includes a small demo script (see below)
reparam - which implements a general class for discrete reparameterizations, as well as particular reparameterizations for categorical and binary random variables.

Implementing a categorical REBAR estimator can be as easy as
```
rep = CategoricalReparam(hard_param) #construct reparameterization of categorical distribution with param as parameter.
grad, var_grad = RELAX(*rep.rebar_params(f, weight=nu), [hard_param], var_params=[nu]) #calculate REBAR and dynamic REBAR gradients for loss f with control variate parameter nu
```

Running the relax module as a script starts a small demo illustrating the advantages of using REBAR

![Result of running the REBAR demo.](rebar_demo.png)

Here, REBAR clearly outperforms the standard score estimator in terms of variance, and optimizing the hyperparameters reduces the variance further.
