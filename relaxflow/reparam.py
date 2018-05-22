import numpy as np
import tensorflow as tf
EPSILON = 1e-16


class DiscreteReparam:
    """An abstract class for implementing reparameterizations of
    discrete random variables.
    Correctly constructs draws from p(z), p(b|z) and p(z|b) for a discrete v
    variable b and its reparameterization z, using tf.stop_gradient where
    appropriate.

    Attributes:
        param: parameter of discrete distribution.
        u: noise variable used to generate z.
        z: reparameterization of random variable b.
        b: discretized random variable, using self.gate(z).
        gatedz: relaxed version of b, using self.softgate(z).
        v: noise variable used to generate zb.
        zb: reparameterization of random variable b, conditioned on observed b.
            i.e. zb~p(z|b).
        gatedzb: relaxed version of b, using self.gate(zb).

    Methods:


    """

    def __init__(self, param, noise=None, coupled=False,
                 cond_noise=None, temperature=1.):
        """Creates an instance of DiscreteReparam.

        Args:
            param: Tensorflow tensor equal to parameter of distribution.
            noise: Tensorflow tensor to be used as noise source u.
            coupled: Boolean indicating whether to use coupled or independent
                noise for z and zb.

        """
        with tf.name_scope("reparameterization"):
            self.param = param #assume unconstrained
            self.noise_shape = param.shape.as_list()
            self.temperature = temperature
            self.coupled = coupled
            
            #set noise tensor u to external tensor
            if noise is not None:
                assert(noise.shape.as_list() == self.noise_shape)
                self.u = noise
            else:
                self.u = tf.random_uniform(self.noise_shape, dtype=param.dtype)
            self.u = tf.stop_gradient(self.u)
            
            #sample random variable b by discretizing random z
            with tf.name_scope("forward"):
                self.z = self.forward(self.param, self.u)
                with tf.name_scope("gate"):
                    self.b = tf.stop_gradient(self.gate(self.z))
                    self.gatedz = self.softgate(self.z, self.temperature)

            #set cond. noise tensor v to external tensor, or couple to u
            with tf.name_scope("cond_noise"):
                if coupled and (cond_noise is not None):
                    raise(ValueError("""coupled and cond_noise
                                     keywords are
                                     mutually exclusive"""))
                elif coupled:
                    self.v = self.coupling(self.param, self.b, self.u)

                elif cond_noise is not None:
                    assert(cond_noise.shape.as_list() == self.noise_shape)
                    self.v = cond_noise
                else:
                    self.v = tf.random_uniform(self.noise_shape,
                                               dtype=param.dtype)
                self.v = tf.stop_gradient(self.v)
            
            #sample a new z, conditional on sampled b
            with tf.name_scope("backward"):
                self.zb = self.backward(self.param, self.b, self.v)
                with tf.name_scope("gate"):
                    self.gatedzb = self.softgate(self.zb, self.temperature)
            self.logp = self.logpdf(self.param, self.b)

    def rebar_params(self, f_loss, weight):
        """Returns parameters for relax.RELAX() function corresponding to a
        standard REBAR estimator.

        Example:
            relax.RELAX(*discretereparam.rebar_params(), hard_params)

        Args:
            f_loss: function object providing the loss the REBAR estimator is
                derived with respect to.
            weight: a scalar parameter used as a scaling factor on the control
                variate.

        Returns:
            rebar_params: a 4-tuple corresponding to the parameters
                (loss, control, conditional_control, logp) expected by the
                RELAX function.
        """
        return (f_loss(self.b),
                weight*f_loss(self.gatedz),
                weight*f_loss(self.gatedzb),
                self.logp)

    @staticmethod
    def forward(param, u):
        raise(NotImplementedError)

    @staticmethod
    def backward(param, b, u):
        raise(NotImplementedError)

    @staticmethod
    def gate(z):
        raise(NotImplementedError)

    @staticmethod
    def softgate(z, t):
        return tf.nn.softmax(z/t, axis=-1)

    @staticmethod
    def coupling(param, b, u):
        raise(NotImplementedError)


class BinaryReparam(DiscreteReparam):
    """DiscreteReparam subclass implementing reparameterization for
        binary ({0,1} Bernoulli) random variables.
        If parameter is of size (N,), it parameterizes N binary variables,
        and the n'th element of the parameter vector param is the logit of the
        probability of the n'th binary variable being 1.
    """
    @staticmethod
    def logpdf(param, b):
        return b * (-tf.nn.softplus(-param)) + (1 - b) * (-param - tf.nn.softplus(-param))

    @staticmethod
    def forward(param, u):
        return binary_forward(param, u)

    @staticmethod
    def backward(param, b, v):
        return binary_backward(param, v, b)

    @staticmethod
    def gate(z):
        return (1.+tf.sign(z))/2.

    @staticmethod
    def coupling(param, b, u):
        uprime = tf.nn.sigmoid(-param)
        v = ((1. - b) * (u/tf.clip_by_value(uprime, EPSILON, 1.)) +
             b * ((u - uprime) / tf.clip_by_value(1.-uprime, EPSILON, 1.)))
        return tf.clip_by_value(v, 0., 1.)


def binary_forward(param, noise=None):
    """draw reparameterization z of binary variable b from p(z)."""
    if noise is not None:
        u = noise
    else:
        u = tf.random_uniform(param.shape.as_list(), dtype=param.dtype)
    z = param + tf.log(u) - tf.log(1. - u)
    return z


def binary_backward(param, b, noise=None):
    """draw reparameterization z of binary variable b from p(z|b)."""
    if noise is not None:
        v = noise
    else:
        v = tf.random_uniform(param.shape.as_list(), dtype=param.dtype)
    uprime = tf.nn.sigmoid(-param)
    ub = b * (uprime + (1. - uprime) * v) + (1. - b) * uprime * v
    ub = tf.clip_by_value(ub, 0., 1.)
    zb = param + tf.log(ub) - tf.log(1. - ub)
    return zb


class CategoricalReparam(DiscreteReparam):
    """DiscreteReparam subclass implementing reparameterization for
        categorical random variables.
        If the parameter param is of size (N, K), it parameterizes N categorical
        variables with an event space of size K. Random draws are one-hot
        encoded so that the random variable is also size (N, K). The n'th row
        of the parameter matrix is the unnormalized log-probabilities of the
        n'th categorical variable, such that tf.nn.softmax(param, axis=-1) gives
        the categorical probabilities.
    """
    @staticmethod
    def logpdf(param, b):
        return tf.reduce_sum(param*b)

    @staticmethod
    def forward(param, u):
        return categorical_forward(param, u)

    @staticmethod
    def backward(param, b, v):
        return categorical_backward(param, b, v)

    @staticmethod
    def gate(z):
        return tf.one_hot(tf.argmax(z, axis=-1), z.shape[-1], dtype=z.dtype)

    @staticmethod
    def coupling(param, b, u):
        def robustgumbelcdf(g, K):
            return tf.exp(EPSILON - tf.exp(-g)*tf.exp(-K)) - EPSILON

        z = param - tf.log(-tf.log(u + EPSILON) + EPSILON,
                       name="gumbel")
        vtop = robustgumbelcdf(z, -tf.reduce_logsumexp(param, axis=-1,
                               keepdims=True))
        topgumbel = tf.reduce_sum(b*z, axis=-1, keepdims=True)
        vrest = tf.exp(-tf.exp(param)*(tf.exp(-z)-tf.exp(-topgumbel)))
        return (1.-b)*vrest + b*vtop


def categorical_forward(param, noise=None):
    """draw reparameterization z of categorical variable b from p(z)."""
    if noise is not None:
        u = noise
    else:
        u = tf.random_uniform(param.shape.as_list(), dtype=param.dtype)
    gumbel = - tf.log(-tf.log(u + EPSILON) + EPSILON, name="gumbel")
    return param + gumbel


def categorical_backward(param, s, noise=None):
    """draw reparameterization z of categorical variable b from p(z|b)."""
    def truncated_gumbel(gumbel, truncation):
        return -tf.log(EPSILON + tf.exp(-gumbel) + tf.exp(-truncation))
    if noise is not None:
        v = noise
    else:
        v = tf.random_uniform(param.shape.as_list(), dtype=param.dtype)

    gumbel = -tf.log(-tf.log(v + EPSILON) + EPSILON, name="gumbel")
    topgumbels = gumbel + tf.reduce_logsumexp(param, axis=-1, keepdims=True)
    topgumbel = tf.reduce_sum(s*topgumbels, axis=-1, keepdims=True)

    truncgumbel = truncated_gumbel(gumbel + param, topgumbel)
    return (1. - s)*truncgumbel + s*topgumbels
