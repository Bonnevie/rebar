import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPSILON = 1e-16

sigma = lambda z, t: tf.nn.softmax(z/t, dim=-1) #scaled softmax
H = lambda z: (1.+tf.sign(z))/2. #Heaviside
select_max = lambda z, K: tf.one_hot(tf.argmax(z, axis=-1), K) #one-hot argmax

class DiscreteReparam:
    def __init__(self, param, noise=None, coupled=False, cond_noise=None):
        with tf.name_scope("reparameterization"):
            self.param = param
            self.noise_shape = param.shape.as_list()
            self.temperature = 1.

            if noise is not None:
                assert(noise.shape.as_list() == self.noise_shape)
                self.u = noise
            else:
                self.u = tf.random_uniform(self.noise_shape, dtype=param.dtype)
            self.u = tf.stop_gradient(self.u)
            with tf.name_scope("forward"):
                self.z = self.forward(self.param, self.u)
                with tf.name_scope("gate"):
                    self.b = tf.stop_gradient(self.gate(self.z))
                    self.gatedz = self.softgate(self.z, self.temperature)
            #use "is not None" to comply with Tensorflow
            with tf.name_scope("cond_noise"):
                if coupled and (cond_noise is not None):
                    raise(ValueError("coupled and cond_noise keywords are mutually exclusive"))
                elif coupled:
                    self.v = self.coupling(self.param, self.b, self.u)
                elif cond_noise is not None:
                    assert(cond_noise.shape.as_list() == self.noise_shape)
                    self.v = cond_noise
                else:
                    self.v = tf.random_uniform(self.noise_shape, dtype=param.dtype)
                self.v = tf.stop_gradient(self.v)

            with tf.name_scope("backward"):
                self.zb = self.backward(self.param, self.b, self.v)
                with tf.name_scope("gate"):
                    self.gatedzb = self.softgate(self.b, self.temperature)
                    self.logp = self.logpdf(self.param, self.b)

    def rebar_params(self, f_loss, weight):
        return (f_loss(self.b),
                weight*f_loss(self.softgate(self.z, self.temperature)),
                weight*f_loss(self.softgate(self.zb, self.temperature)),
                self.logp)

    @staticmethod
    def forward(p, u):
        raise(NotImplementedError)

    @staticmethod
    def backward(p, b, u):
        raise(NotImplementedError)

    @staticmethod
    def gate(z):
        raise(NotImplementedError)

    @staticmethod
    def softgate(z, t):
        return tf.nn.softmax(z/t, dim=-1)

    @staticmethod
    def coupling(p, b, u):
        raise(NotImplementedError)


class BinaryReparam(DiscreteReparam):
    @staticmethod
    def logpdf(p, b):
        return tf.reduce_sum(b*tf.log(p) +  (1.-b)*tf.log(1.-p))

    @staticmethod
    def forward(p, u):
        return binary_forward(p, u)

    @staticmethod
    def backward(p, b, v):
        return binary_backward(p, v, b)

    @staticmethod
    def gate(z):
        return (1.+tf.sign(z))/2.

    @staticmethod
    def coupling(p, b, u):
        uprime = 1. - p
        return (1. - b) * (u/uprime) + b * ((u - uprime) / (1. - uprime))

def binary_forward(p, noise=None):
    '''draw reparameterization z of binary variable b from p(z).'''
    if noise is not None:
        u = noise
    else:
        u = tf.random_uniform(p.shape.as_list(), dtype=p.dtype)
    z = tf.log(p) - tf.log(1. - p) + tf.log(u) - tf.log(1. - u)
    return z

def binary_backward(p, b, noise=None):
    '''draw reparameterization z of binary variable b from p(z|b).'''
    if noise is not None:
        v = noise
    else:
        v = tf.random_uniform(p.shape.as_list(), dtype=p.dtype)
    uprime = 1. - p
    ub = b * (uprime + (1.- uprime) * v) + (1.-b) * uprime * v #v * (b * (1. - p) + (1. - b) * p)
    zb = tf.log(p) - tf.log(1. - p) + tf.log(ub) - tf.log(1. - ub)
    return zb

class CategoricalReparam(DiscreteReparam):
    @staticmethod
    def logpdf(p, b):
        return tf.reduce_sum(p*b)

    @staticmethod
    def forward(p, u):
        return categorical_forward(p, u)

    @staticmethod
    def backward(p, b, v):
        return categorical_backward(p, b, v)

    @staticmethod
    def gate(z):
        return tf.one_hot(tf.argmax(z, axis=-1), z.shape[-1], dtype=z.dtype)

    @staticmethod
    def coupling(p, b, u):
        def gumbelcdf(g):
            return tf.exp(EPSILON - tf.exp(-g)) - EPSILON
        def robustgumbelcdf(g, K):
            return tf.exp(EPSILON - tf.exp(-g)*tf.exp(-K)) - EPSILON

        z = p - tf.log( - tf.log(u + EPSILON) + EPSILON , name="gumbel")
        vtop = robustgumbelcdf(z, -tf.reduce_logsumexp(p, axis=-1, keep_dims=True))
        topgumbel = tf.reduce_sum(b*z, axis=-1, keep_dims=True)
        vrest = tf.exp(-tf.exp(p)*(tf.exp(-z)-tf.exp(-topgumbel)))
        #vrest = gumbelcdf(-(p + tf.log(EPSILON + tf.exp(-z) - tf.exp(-topgumbel))))
        return (1.-b)*vrest + b*vtop

def categorical_forward(alpha, noise=None):
    '''draw reparameterization z of categorical variable b from p(z).'''
    if noise is not None:
        u = noise
    else:
        u = tf.random_uniform(alpha.shape.as_list(), dtype=alpha.dtype)
    gumbel = - tf.log( - tf.log(u + EPSILON) + EPSILON , name="gumbel")
    return alpha + gumbel

def categorical_backward(alpha, s, noise=None):
    '''draw reparameterization z of categorical variable b from p(z|b).'''
    def truncated_gumbel(gumbel, truncation):
        return -tf.log(EPSILON + tf.exp(-gumbel) + tf.exp(-truncation))
    if noise is not None:
        v = noise
    else:
        v = tf.random_uniform(alpha.shape.as_list(), dtype=alpha.dtype)

    gumbel = - tf.log( - tf.log(v + EPSILON) + EPSILON , name="gumbel")
    topgumbels = gumbel + tf.reduce_logsumexp(alpha, axis=-1, keep_dims=True)
    topgumbel = tf.reduce_sum(s*topgumbels, axis=-1, keep_dims=True)

    truncgumbel = truncated_gumbel(gumbel + alpha, topgumbel)
    #return s*truncgumbel + (1.-s)*topgumbels
    return (1.-s)*truncgumbel + s*topgumbels

def buildcontrol(f_loss, f_control, logpdf,
                 forward, backward,
                 hard_gate, soft_gate):
    '''
    Draw z~p(z), and zb~p(z|b) where b=hard_gate(z), and construct loss and
    control variates appropriate for use in RELAX estimator.

    f_loss - Function. Function of discrete variable b.
    f_control - Function. Parametric control variate.
    lnpdf - a function that evaluates the log-probability ln p(b) of the
        discrete random variable b.
    forward - a stochastic TF tensor depending on hard_params.
        Corresponds to z, the reparameterized discrete random variable
        prior to discretization.
    backward - a function that returns a stochastic TF tensor given b.
        Returns zb, the reparameterized discrete random variable
        prior to discretization, conditioned on the discretization being equal
        to the passed discrete variable.
    hard_gate - function that discretizes the latent discrete variable.
    soft_gate - a continuous differentiable relaxation of hard_gate.

    Returns:
        loss - TF tensor f_loss(hard_gate(z)).
        control - TF tensor f_control(soft_gate(z)).
        conditional_control - TF tensor f_control(soft_gate(zb)).
        logp - TF tensor logpdf(b)
    '''
    #forward+backward
    z = forward
    b = tf.stop_gradient(hard_gate(z))
    zb = backward(b)

    loss = f_loss(b)
    control = f_control(soft_gate(z))
    conditional_control = f_control(soft_gate(zb))
    logp = distribution(b)

    return loss, control, conditional_control, logp

def RELAX(loss, control, conditional_control, logp,
          hard_params, params=[], var_params=[]):
    '''Estimate the gradient of "loss" with respect to "hard_params" which
    enter the loss through a stochastic non-differentiable map.
    Use RELAX estimator for the gradient with respect to "hard_params" and
    use a standard sampling estimator for parameters in "params".

    The RELAX estimator estimates the gradient of f(b) where
    H(b)=z and z~p(z;hard_params) using a control variate function c().
    The input "control" corresponds to c(z) and "conditional_control" to c(zb)
    where zb~p(z|b). For a canonical construction, see "buildcontrol" function.
    Note: RELAX estimator may be wrong if construction is not correct.
    Respect above relationship between "control" and "conditional_control" and
    use tf.stop_gradient on the discrete variable as in "buildcontrol".

    loss - scalar tensor to compute gradients for.
    control - differentiable tensor approximating loss.
    conditional_control - identical to control, but with noise conditioned on
        discrete sample.
    logp - a stochastic tensor equal to the log-probability ln p(b)
        of the discrete random variable b being sampled.
    params - other parameters. Uses same samples as REBAR.
    var_params - parameters that the estimator's variance should be minimized for.

    Returns:
        gradients - REBAR gradient estimator for params and hard_params.
        loss - function evaluated in discrete random sample.
        var_grad - gradient of estimator variance for var_params.
    '''
    with tf.name_scope("RELAX"):
        with tf.name_scope("score"):
            scores = tf.gradients(logp, hard_params)

        with tf.name_scope("pure_grad"):
            #compute gradient of loss outside of dependence through b.
            pure_grads = tf.gradients(loss, hard_params)

        with tf.name_scope("relax_grad"):
            #Derivative of differentiable control variate.
            relax_grads = tf.gradients(control - conditional_control, hard_params)

        gradcollection = zip(pure_grads, relax_grads, hard_params, scores)
        hard_params_grad = []

        with tf.name_scope("collect_grads"):
            for pure_grad, relax_grad, hard_param, score in gradcollection:
                #aggregate gradient components
                full_grad = tf.zeros_like(hard_param)
                if pure_grad is not None:
                    full_grad += pure_grad[0]
                if relax_grad is not None:
                    #complete RELAX estimator
                    full_grad += ((loss - conditional_control) * score  +  relax_grad)
                hard_params_grad += [(full_grad, hard_param)]
        with tf.name_scope("auxiliary_grad"):
            #auxiliary gradients
            params_grad = list(zip(tf.gradients(loss, params), params))
            var_params_grad = list(zip(tf.gradients(full_grad, var_params), var_params))

        with tf.name_scope("variance_grad"):
            #compute variance gradient
            var_grad = [(tf.reduce_sum(2*full_grad*grad), param) for grad, param in var_params_grad]
        return hard_params_grad + params_grad, var_grad

def REBAR(f, hard_params, forward, backward, distribution,
          hard_gate, soft_gate, nu=1., params=[], var_params=[]):
    '''Estimate the gradient of the composed function:

        f(hard_gate(forward)), params)

    composed of,

        f(b, params) - function in discrete variable b and parameters param.
        b=hard_gate(z) - discretization of z.
        z=forward - differentiable reparameterization of b,
                    stochastic function of hard_params.

    using the REBAR estimator for the gradient with respect to hard_params and
    using a standard sampling estimator for params.

    REBAR corresponds to a RELAX estimator where the loss f(hard_gate(z)) is
    estimated by the control f(soft_gate(z)).

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

    #compute canonical control variate
    loss, control, conditional_control, logp = buildcontrol(f, f, distribution,
                                                            forward, backward,
                                                            hard_gate,
                                                            soft_gate)
    #redirect to RELAX
    return RELAX(loss, nu*control, nu*conditional_control, logp,
                 hard_params, params, var_params)

if __name__ is "__main__":

    categorical = False
    N = 20
    K = 10
    R = 10000 #sample repeats
    true_index = np.eye(K)[np.random.randint(0, K, N)]

    # Cost function with 1 good configuration
    def f(z):
        return -tf.square(tf.reduce_sum(z) + tf.reduce_sum(z*true_index))

    Z = tf.identity(tf.Variable(true_index, dtype=tf.float32))
    temp_var = tf.Variable(1.)
    temp = tf.nn.softplus(temp_var)
    nu_var = tf.Variable(1.)
    nu_switch = tf.Variable(1.)
    nu = nu_switch*tf.nn.softplus(nu_var)

    tempc_var = tf.Variable(1.)
    tempc = tf.nn.softplus(tempc_var)
    nuc_var = tf.Variable(1.)
    nuc_switch = tf.Variable(1.)
    nuc = nu_switch*tf.nn.softplus(nuc_var)

    forward = categorical_forward(Z)
    backward = lambda b: categorical_backward(Z, b)
    distribution = lambda b: tf.reduce_sum(b*Z, axis=-1)

    if categorical:
        rep = CategoricalReparam(Z)
    else:
        rep = BinaryReparam(1./(1.+tf.exp(-Z)))
    grad, var_grad = RELAX(f(rep.b), nu*f(sigma(rep.z, temp)),
                                 nu*f(sigma(rep.zb, temp)), rep.logp, [Z], var_params=[temp_var, nu_var])
    if categorical:
        repc = CategoricalReparam(Z, coupled=True)
    else:
        repc = BinaryReparam(1./(1.+tf.exp(-Z)), coupled=True)
    gradc, varc_grad = RELAX(f(repc.b), nuc*f(sigma(repc.z, tempc)),
                                 nuc*f(sigma(repc.zb, tempc)), repc.logp, [Z], var_params=[tempc_var, nuc_var])


    if False:
        grad, loss, var_grad = REBAR(f, [Z], forward, backward, distribution, nu=nu,
                                     hard_gate=lambda z: select_max(z, K),
                                     soft_gate=lambda z: sigma(z, temp),
                                     var_params=[temp_var, nu_var])

    grad_estimator = tf.expand_dims(grad[0][0], 0)
    gradc_estimator = tf.expand_dims(gradc[0][0], 0)
    tf.train.AdamOptimizer()
    opt = tf.train.AdamOptimizer(0.00001)
    train_step = opt.apply_gradients(var_grad)
    optc = tf.train.AdamOptimizer(0.00001)
    trainc_step = opt.apply_gradients(varc_grad)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    #Calculate gradients using REBAR
    base_grads = []
    for _ in range(R):
        base_grads += sess.run([grad_estimator])
    base_grad = np.concatenate(base_grads, axis=0)
    base_mu = base_grad.mean(axis=0)
    base_var = np.square(base_grad - base_mu).mean(axis=0)

    #Calculate gradients using REBAR with coupling
    couple_grads = []
    for _ in range(R):
        couple_grads += sess.run([gradc_estimator])
    couple_grad = np.concatenate(couple_grads, axis=0)
    couple_mu = couple_grad.mean(axis=0)
    couple_var = np.square(couple_grad - couple_mu).mean(axis=0)


    #POptimize nu and temp, then Calculate gradients using REBAR
    for _ in range(20000):
        sess.run([train_step, trainc_step])

    print("optimal temperature: {}\n optimal control weight: {}".format(temp.eval(), nu.eval()))
    opt_grads = []
    for _ in range(R):
        opt_grads += sess.run([grad_estimator])
    opt_grad = np.concatenate(opt_grads, axis=0)
    opt_mu = opt_grad.mean(axis=0)
    opt_var = np.square(opt_grad - opt_mu).mean(axis=0)

    print("optimal temperature (coupled): {}\n optimal control weight (coupled): {}".format(temp.eval(), nu.eval()))
    optc_grads = []
    for _ in range(R):
        optc_grads += sess.run([gradc_estimator])
    optc_grad = np.concatenate(optc_grads, axis=0)
    optc_mu = optc_grad.mean(axis=0)
    optc_var = np.square(optc_grad - optc_mu).mean(axis=0)

    #Calculate gradients without REBAR
    sess.run(tf.assign(nu_switch, 0.))
    raw_grads = []
    for _ in range(R):
        raw_grads += sess.run([grad_estimator])
    raw_grad = np.concatenate(raw_grads, axis=0)
    raw_mu = raw_grad.mean(axis=0)
    raw_var = np.square(raw_grad - raw_mu).mean(axis=0)

    svars = np.column_stack([raw_var.ravel(), base_var.ravel(), couple_var.ravel(), opt_var.ravel(), optc_var.ravel()])
    plt.boxplot(np.log(svars))
    plt.xticks(np.arange(1,6), ['Score Estimator', 'REBAR', 'REBAR (coupled)', 'Optimized REBAR', 'optimized+coupled'])
    plt.ylabel("Log Sample Variance ({} samples)".format(R))
