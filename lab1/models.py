import math
from typing import Union, Collection

import tensorflow as tf

TFData = Union[tf.Tensor, tf.Variable, float]


class GMModel:
    def __init__(self, K):
        self.K = K
        self.mean = tf.Variable(tf.random.normal(shape=[K]))
        self.logvar = tf.Variable(tf.random.normal(shape=[K]))
        self.logpi = tf.Variable(tf.zeros(shape=[K]))

    @property
    def variables(self) -> Collection[TFData]:
        return self.mean, self.logvar, self.logpi

    @staticmethod
    def neglog_normal_pdf(x: TFData, mean: TFData, logvar: TFData):
        var = tf.exp(logvar)

        return 0.5 * (tf.math.log(2 * math.pi) + logvar + (x - mean) ** 2 / var)

    @tf.function
    def loss(self, data: TFData):
        data = tf.cast(tf.reshape(data, [-1, 1]), self.mean.dtype)
        component_nll = self.neglog_normal_pdf(data, self.mean, self.logvar)
        component_nll -= tf.nn.log_softmax(self.logpi)

        return -tf.reduce_logsumexp(-component_nll, axis=1)

    def p_xz(self, x: TFData, k: int) -> TFData:
        x = tf.cast(tf.convert_to_tensor(x), self.mean.dtype)

        return tf.exp(-self.neglog_normal_pdf(x, self.mean[k], self.logvar[k]))

    def p_x(self, x: TFData) -> TFData:
        x = tf.cast(tf.reshape(x, [-1, 1]), self.mean.dtype)
        component_log_prob = -self.neglog_normal_pdf(x, self.mean, self.logvar)
        log_prob = tf.reduce_logsumexp(
            tf.nn.log_softmax(self.logpi) + component_log_prob,
            axis=1,
        )

        return tf.exp(log_prob)
