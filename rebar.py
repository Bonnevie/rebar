import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPSILON = 1e-16

sigma = lambda z, t: tf.nn.softmax(z/t, dim=-1) #scaled softmax
H = lambda z: (1.+tf.sign(z))/2. #Heaviside
select_max = lambda z, K: tf.one_hot(tf.argmax(z, axis=-1), K) #one-hot argmax
#score function of Bernoulli using raw probabilities p
binary_score = lambda b, p: (b / p - (1. - b) / (1. - p))
#score function of categorical using log probabilities alpha
discrete_score = lambda b, alpha: b

def binary_forward(p):
    '''draw reparameterization z of binary variable b from p(z).'''
    u = tf.random_uniform(p.shape.as_list(), dtype=p.dtype)
    z = tf.log(p) - tf.log(1. - p) + tf.log(u) - tf.log(1. - u)
    return z

def binary_backward(p, b):
    '''draw reparameterization z of binary variable b from p(z|b).'''
    v = tf.random_uniform(p.shape.as_list(), dtype=p.dtype)
    ub = b * p + v * (b * (1. - p) + (1. - b) * p)
    zb = tf.log(p) - tf.log(1. - p) + tf.log(ub) - tf.log(1. - ub)
    return zb

def categorical_forward(alpha):
    '''draw reparameterization z of categorical variable b from p(z).'''
    u = tf.random_uniform(alpha.shape.as_list(), dtype=alpha.dtype)
    gumbel = - tf.log( - tf.log(u + EPSILON) + EPSILON , name="gumbel")
    return alpha + gumbel

def categorical_backward(alpha, s):
    '''draw reparameterization z of categorical variable b from p(z|b).'''
    def truncated_gumbel(gumbel, truncation):
        return -tf.log(tf.exp(-gumbel) + tf.exp(-truncation))

    v = tf.random_uniform(alpha.shape.as_list(), dtype=alpha.dtype)
    gumbel = - tf.log( - tf.log(v + EPSILON) + EPSILON , name="gumbel")
    topgumbels = gumbel + tf.reduce_logsumexp(alpha, axis=-1, keep_dims=True)
    topgumbel = tf.reduce_sum(s*topgumbels, axis=-1, keep_dims=True)

    truncgumbel = truncated_gumbel(gumbel + alpha, topgumbel)
    return (1.-s)*truncgumbel + s*topgumbels

def REBAR(f, hard_params, forward, backward, distribution,
          hard_gate, soft_gate, nu=1., params=[], var_params=[]):
    '''Estimate the gradient of the composed function:

        f(hard_gate(forward(preprocess(hard_params))), params)

    composed of,

        f(b, params) - function in discrete variable b and parameters param.
        b=hard_gate(z) - discretization of z.
        z=forward - differentiable reparameterization.

    using the REBAR estimator for the gradient with respect to hard_params and
    using a standard sampling estimator for params.

    f - function of interest.
    hard_params - parameter that codes for discrete random variables.
    forward - a stochastic TF tensor depending on hard_params.
        Corresponds to the reparameterized discrete random variable
        prior to discretization.
    backward - a function that returns a stochastic TF tensor given b.
        Returns the reparameterized discrete random variable
        prior to discretization, conditioned on the discretization being equal
        to the passed discrete variable.
    distribution - a function that evaluates the log-probability of the
        discrete random variable being sampled.
    hard_gate - function that discretizes the latent discrete variable.
    soft_gate - a continuous differentiable relaxation of hard_gate.
    params - other parameters. Uses same samples as REBAR.
    var_params - parameters that the estimator's variance should be minimized for.
    nu - the REBAR regularization weight.

    Returns:
        gradients - REBAR gradient estimator for params and hard_params.
        loss - function evaluated in discrete random sample.
        var_grad - gradient of estimator variance for var_params.
    '''

    #forward+backward
    z = forward
    b = tf.stop_gradient(hard_gate(z))
    zb = backward(b)

    #score
    score = tf.gradients(distribution(b), hard_params)[0]

    #gradient collection
    pure_grad = tf.gradients(f(b), hard_params)[0]
    rebar_grad = tf.gradients(f(soft_gate(z)) - f(soft_gate(zb)), hard_params)[0]
    full_grad = tf.zeros_like(hard_params)
    if pure_grad is not None:
        full_grad += pure_grad
    if rebar_grad is not None:
        full_grad += ((f(b) - nu * f(soft_gate(zb))) * score +
                                     nu * rebar_grad)

    params_grad = list(zip(tf.gradients(f(b), params), params))
    var_params_grad = list(zip(tf.gradients(full_grad, var_params), var_params))

    var_grad = [(tf.reduce_sum(2*full_grad*grad), param) for grad, param in var_params_grad]
    return [(full_grad, hard_params)] + params_grad, f(b), var_grad




if __name__ is "__main__":

    N = 20
    K = 10
    R = 10000 #sample repeats
    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.reduce_sum(z) - tf.reduce_sum(z*true_index)

    Z = tf.Variable(true_index, dtype=tf.float32)
    temp_var = tf.Variable(1.)
    temp = tf.nn.softplus(temp_var)
    nu_var = tf.Variable(1.)
    nu_switch = tf.Variable(1.)
    nu = nu_switch*tf.nn.softplus(nu_var)

    forward = categorical_forward(Z)
    backward = lambda b: categorical_backward(Z, b)
    distribution = lambda b: tf.reduce_sum(b*Z, axis=-1)

    grad, loss, var_grad = REBAR(f, Z, forward, backward, distribution, nu=nu,
                                 hard_gate=lambda z: select_max(z, K),
                                 soft_gate=lambda z: sigma(z, temp),
                                 var_params=[temp_var, nu_var])

    grad_estimator = grad[0]

    opt = tf.train.AdamOptimizer()
    train_step = opt.apply_gradients(var_grad)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    #Calculate gradients using REBAR
    base_grads = []
    for _ in range(R):
        base_grads += sess.run([grad_estimator])
    base_grad = np.concatenate(base_grads, axis=0)
    base_mu = base_grad.mean(axis=0)
    base_var = np.square(base_grad - base_mu).mean(axis=0)

    #POptimize nu and temp, then Calculate gradients using REBAR
    for _ in range(20000):
        sess.run(train_step)
    print("optimal temperature: {}\n optimal control weight: {}".format(temp.eval(), nu.eval()))
    opt_grads = []
    for _ in range(R):
        opt_grads += sess.run([grad_estimator])
    opt_grad = np.concatenate(opt_grads, axis=0)
    opt_mu = opt_grad.mean(axis=0)
    opt_var = np.square(opt_grad - opt_mu).mean(axis=0)

    #Calculate gradients without REBAR
    sess.run(tf.assign(nu_switch, 0.))
    raw_grads = []
    for _ in range(R):
        raw_grads += sess.run([grad_estimator])
    raw_grad = np.concatenate(raw_grads, axis=0)
    raw_mu = raw_grad.mean(axis=0)
    raw_var = np.square(raw_grad - raw_mu).mean(axis=0)

    svars = np.column_stack([base_var.ravel(), opt_var.ravel(), raw_var.ravel()])
    plt.boxplot(np.log(svars))
    plt.xticks(np.arange(1,4), ['REBAR', 'Optimized REBAR', 'Score Estimator'])
    plt.ylabel("Sample Variance ({} samples)".format(R))
