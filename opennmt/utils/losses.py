"""Define losses."""

import tensorflow as tf
from opennmt.utils.misc import add_dict_to_collection


def _smooth_one_hot_labels(logits, labels, label_smoothing):
  label_smoothing = tf.constant(label_smoothing, dtype=logits.dtype)
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
      tf.cast(labels, tf.int32),
      num_classes,
      on_value=1.0 - label_smoothing,
      off_value=label_smoothing / tf.cast(num_classes - 1, label_smoothing.dtype),
      dtype=logits.dtype)

def _softmax_cross_entropy(logits, labels, label_smoothing, mode):
  # Computes the softmax in full precision.
  if logits.dtype.base_dtype != tf.float32:
    logits = tf.cast(logits, tf.float32)
  if mode == tf.estimator.ModeKeys.TRAIN and label_smoothing > 0.0:
    smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
    if hasattr(tf.nn, "softmax_cross_entropy_with_logits_v2"):
      cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits_v2
    else:
      cross_entropy_fn = tf.nn.softmax_cross_entropy_with_logits
    return cross_entropy_fn(
        logits=logits, labels=smoothed_labels)
  else:
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

def cross_entropy_sequence_loss(logits,
                                labels,
                                sequence_length,
                                label_smoothing=0.0,
                                average_in_time=False,
                                mode=tf.estimator.ModeKeys.TRAIN,
                                master_mask=None):
  """Computes the cross entropy loss of sequences.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    sequence_length: The length of each sequence.
    label_smoothing: The label smoothing value.
    average_in_time: If ``True``, also average the loss in the time dimension.
    mode: A ``tf.estimator.ModeKeys`` mode.

  Returns:
    A tuple (cumulated loss, loss normalizer, token-level normalizer).
  """
  tf.logging.info(" >> [losses.py cross_entropy_sequence_loss] average_in_time = {}".format(average_in_time))
  tf.logging.info(" >> [losses.py cross_entropy_sequence_loss] logits = {}".format(logits))
  tf.logging.info(" >> [losses.py cross_entropy_sequence_loss] labels = {}".format(labels))
  tf.logging.info(" >> [losses.py cross_entropy_sequence_loss] sequence_length = {}".format(sequence_length))

  batch_size = tf.shape(logits)[0]
  max_time = tf.shape(logits)[-2]

  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, mode)
  weights = tf.sequence_mask(sequence_length, maxlen=max_time, dtype=cross_entropy.dtype)

  if master_mask is not None:
    add_dict_to_collection("debug", {"master_mask": master_mask})
    weights = weights * master_mask

  loss = tf.reduce_sum(cross_entropy * weights)
  loss_token_normalizer = tf.reduce_sum(weights)

  add_dict_to_collection("debug", {"batch_size": batch_size,
                                   "max_time": max_time,
                                   # "logits": logits,
                                   "logits_shape": tf.shape(logits),
                                   # "labels": labels,
                                   "labels_shape": tf.shape(labels),
                                   "cross_entropy": cross_entropy,
                                   "cross_entropy_shape": tf.shape(cross_entropy),
                                   "weights": weights,
                                   "loss": loss,
                                   "loss_token_normalizer": loss_token_normalizer,
                                   "weights_shape": tf.shape(weights),
                                   })

  if average_in_time or mode != tf.estimator.ModeKeys.TRAIN:
    loss_normalizer = loss_token_normalizer
  else:
    loss_normalizer = tf.cast(batch_size, loss.dtype)

  return loss, loss_normalizer, loss_token_normalizer

def cross_entropy_loss(logits,
                       labels,
                       label_smoothing=0.0,
                       mode=tf.estimator.ModeKeys.TRAIN):
  """Computes the cross entropy loss.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    label_smoothing: The label smoothing value.
    mode: A ``tf.estimator.ModeKeys`` mode.

  Returns:
    The cumulated loss and the loss normalizer.
  """
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, mode)
  loss = tf.reduce_sum(cross_entropy)
  loss_normalizer = tf.cast(tf.shape(cross_entropy)[0], loss.dtype)
  return loss, loss_normalizer
