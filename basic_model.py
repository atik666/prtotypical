from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import tensorflow as tf

from fewshot.models.kmeans_utils import compute_logits
from fewshot.models.model import Model
from fewshot.models.model_factory import RegisterModel
from fewshot.models.nnlib import (concat, weight_variable)
from fewshot.utils import logger
from fewshot.utils.debug import debug_identity

FLAGS = tf.flags.FLAGS
log = logger.get()


@RegisterModel("basic")
class BasicModel(Model):
  """A basic retrieval model that runs the images through a CNN and compute
  basic similarity scores."""

  def get_encoded_inputs(self, *x_list, **kwargs):
    """Runs the reference and candidate images through the feature model phi.
    Returns:
      h_train: [B, N, D]
      h_unlabel: [B, P, D]
      h_test: [B, M, D]
    """
    config = self.config
    bsize = tf.shape(self.x_train)[0]
    bsize = tf.shape(x_list[0])[0]
    num = [tf.shape(xx)[1] for xx in x_list]
    x_all = concat(x_list, 1)
    if 'ext_wts' in kwargs:
      ext_wts = kwargs['ext_wts']
    else:
      ext_wts = None
    x_all = tf.reshape(x_all,
                       [-1, config.height, config.width, config.num_channel])
    h_all = self.phi(x_all, ext_wts=ext_wts)
    h_all = tf.reshape(h_all, [bsize, sum(num), -1])
    h_list = tf.split(h_all, num, axis=1)
    return h_list

  def _compute_protos(self, nclasses, h_train, y_train):
    """Computes the prototypes, cluster centers.
    Args:
      nclasses: Int. Number of classes.
      h_train: [B, N, D], Train features.
      y_train: [B, N], Train class labels.
    Returns:
      protos: [B, K, D], Test prediction.
    """
    protos = [None] * nclasses
    for kk in range(nclasses):
      # [B, N, 1]
      ksel = tf.expand_dims(tf.cast(tf.equal(y_train, kk), h_train.dtype), 2)
      # [B, N, D]
      protos[kk] = tf.reduce_sum(h_train * ksel, [1], keep_dims=True)
      protos[kk] /= tf.reduce_sum(ksel, [1, 2], keep_dims=True)
      protos[kk] = debug_identity(protos[kk], "proto")
    protos = concat(protos, 1)  # [B, K, D]
    return protos

  def predict(self):
    """See `model.py` for documentation."""
    h_train, h_test = self.get_encoded_inputs(self.x_train, self.x_test)
    y_train = self.y_train
    nclasses = self.nway
    protos = self._compute_protos(nclasses, h_train, y_train)
    logits = compute_logits(protos, h_test)
    return [logits]

  def get_train_op(self, logits, y_test):
    """See `model.py` for documentation."""
    if FLAGS.allstep:
      log.info("Compute average loss for all timestep.")
      if self.nway > 1:
        loss = tf.add_n([
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=ll, labels=y_test) for ll in logits
        ]) / float(len(logits))
      else:
        loss = tf.add_n([
            tf.nn.sigmoid_cross_entropy_with_logits(logits=ll, labels=y_test)
            for ll in logits
        ]) / float(len(logits))
    else:
      log.info("Compute loss for the final timestep.")
      if self.nway > 1:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits[-1], labels=y_test)
      else:
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits[-1], labels=y_test)
    loss = tf.reduce_mean(loss)
    wd_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    log.info("Weight decay variables: {}".format(wd_losses))
    if len(wd_losses) > 0:
      loss += tf.add_n(wd_losses)
    opt = tf.train.AdamOptimizer(self.learn_rate)
    grads_and_vars = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads_and_vars)
    return loss, train_op