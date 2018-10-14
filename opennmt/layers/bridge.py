"""Define bridges: logic of passing the encoder state to the decoder."""

import abc
import six

import tensorflow as tf
from opennmt.utils.misc import add_dict_to_collection


def assert_state_is_compatible(expected_state, state):
  """Asserts that states are compatible.

  Args:
    expected_state: The reference state.
    state: The state that must be compatible with :obj:`expected_state`.

  Raises:
    ValueError: if the states are incompatible.
  """
  # Check structure compatibility.
  tf.contrib.framework.nest.assert_same_structure(expected_state, state)

  # Check shape compatibility.
  expected_state_flat = tf.contrib.framework.nest.flatten(expected_state)
  state_flat = tf.contrib.framework.nest.flatten(state)

  tf.logging.info(" >> [bridge.py class assert_state_is_compatible()] expected_state_flat = {}".format(expected_state_flat))
  tf.logging.info(" >> [bridge.py class assert_state_is_compatible()] state_flat = {}".format(state_flat))

  for x, y in zip(expected_state_flat, state_flat):
    if tf.contrib.framework.is_tensor(x):
      tf.logging.info(" >> [bridge.py class assert_state_is_compatible()] x = {}".format(x))
      tf.logging.info(" >> [bridge.py class assert_state_is_compatible()] y = {}".format(y))
      tf.contrib.framework.with_same_shape(x, y)


@six.add_metaclass(abc.ABCMeta)
class Bridge(object):
  """Base class for bridges."""

  def __call__(self, encoder_state, decoder_zero_state):
    """Returns the initial decoder state.

    Args:
      encoder_state: The encoder state.
      decoder_zero_state: The default decoder state.

    Returns:
      The decoder initial state.
    """
    return self._build(encoder_state, decoder_zero_state)

  @abc.abstractmethod
  def _build(self, encoder_state, decoder_zero_state):
    raise NotImplementedError()


class CopyBridge(Bridge):
  """A bridge that passes the encoder state as is."""

  def _build(self, encoder_state, decoder_zero_state):
    tf.logging.info(" >> [bridge.py class CopyBridge _build] encoder_state = {}".format(encoder_state))
    tf.logging.info(" >> [bridge.py class CopyBridge _build] decoder_zero_state = {}".format(decoder_zero_state))
    assert_state_is_compatible(decoder_zero_state, encoder_state)
    return encoder_state


class ZeroBridge(Bridge):
  """A bridge that does not pass information from the encoder."""

  def _build(self, encoder_state, decoder_zero_state):
    # Simply return the default decoder state.
    return decoder_zero_state


class DenseBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    # Flattened states.
    tf.logging.info(" >> [bridge.py class DenseBridge _build] encoder_state = {}".format(encoder_state))
    tf.logging.info(" >> [bridge.py class DenseBridge _build] decoder_zero_state = {}".format(decoder_zero_state))

    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(
      " >> [bridge.py class DenseBridge _build] encoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(
      " >> [bridge.py class DenseBridge _build] decoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    encoder_state_concat = tf.concat(encoder_state_flat, 1)

    # Extract decoder state sizes.
    decoder_state_size = []
    for tensor in decoder_state_flat:
      decoder_state_size.append(tensor.get_shape().as_list()[-1])

    decoder_total_size = sum(decoder_state_size)

    # Apply linear transformation.
    transformed = tf.layers.dense(
        encoder_state_concat,
        decoder_total_size,
        activation=self.activation,
        name="dense_bridge")
    tf.logging.info(" >> [bridge.py class DenseBridge _build] transformed = \n{}".format(transformed))

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)

    # Pack as the origial decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, splitted)

class AttentionWrapperStateAggregatedDenseBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    '''
    AttentionWrapperState(
      cell_state=LSTMStateTuple(
        c=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/lstm_cell/add_1:0' shape=(?, 128) dtype=float32>, 
        h=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/lstm_cell/mul_2:0' shape=(?, 128) dtype=float32>), 
      attention=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/concat_2:0' shape=(?, 128) dtype=float32>, 
      time=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/add:0' shape=() dtype=int32>, 
      alignments=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/Softmax:0' shape=(?, ?) dtype=float32>, alignment_history=<tensorflow.python.util.tf_should_use.TFShouldUseWarningWrapper object at 0x7f35d43e2bd0>, 
      attention_state=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/Softmax:0' shape=(?, ?) dtype=float32>)
    
    By LuongAttention, attention_state = alignments
    '''

    for s in [encoder_state, decoder_zero_state]:
      if not isinstance(s, tf.contrib.seq2seq.AttentionWrapperState):
        raise ValueError(
          "AttentionWrapperStateAggregatedDenseBridge only supports linear transformation on AttentionWrapperState states")

    # Flattened states.
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] encoder_state = {}"
      .format(encoder_state))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] decoder_zero_state = {}"
      .format(decoder_zero_state))

    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(" >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] encoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(" >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] decoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    encoder_state_concat = tf.concat(encoder_state_flat[:3], 1) # (c_m, h_m, a_m)
    decoder_state_concat = tf.concat(decoder_state_flat[:3], 1) # (c_s, h_s, a_s)
    aggragated_state_concat = tf.concat([encoder_state_concat, decoder_state_concat], 1)

    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] encoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] decoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] aggragated_state_concat = {}"
      .format(aggragated_state_concat))

    # Extract decoder state sizes.
    decoder_state_size = [x.get_shape().as_list()[-1] for x in decoder_state_flat[:3]]
    decoder_total_size = sum(decoder_state_size)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] decoder_state_size = {}"
      .format(decoder_state_size))

    # Apply linear transformation.
    # TODO: maybe not using bias is better
    transformed = tf.layers.dense(
        aggragated_state_concat,
        decoder_total_size,
        activation=self.activation,
        name="AttentionWrapperStateAggregatedDenseBridge")

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] splitted = {}".format(splitted))
    combined = splitted + decoder_state_flat[3:]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, combined)

class AttentionWrapperStatePairwiseDenseBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """

    for s in [encoder_state, decoder_zero_state]:
      if not isinstance(s, tf.contrib.seq2seq.AttentionWrapperState):
        raise ValueError(
          "AttentionWrapperStatePairwiseDenseBridge only supports linear transformation on AttentionWrapperState states")

    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] encoder_state = {}"
      .format(encoder_state))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] decoder_zero_state = {}"
      .format(decoder_zero_state))

    # Flattened states.
    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] encoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] decoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    # Extract decoder state size
      # decoder state sizes for h,c,a should be the same [batch, depth]
    decoder_state_size = decoder_state_flat[0].get_shape().as_list()[-1]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] decoder_state_size = {}"
      .format(decoder_state_size))

    # Apply linear transformation on [c_m, c_s], [h_m, h_s], [a_m, a_s] respectively
    # TODO: maybe not using bias is better
    transformed = []
    for i in range(3):
      vec_in = tf.concat([encoder_state_flat[i], decoder_state_flat[i]], 1)
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] vec_in = {}".format(vec_in))
      projected = tf.layers.dense(vec_in,
                                  decoder_state_size,
                                  activation=self.activation,
                                  name="AWSPairwiseDenseBridge_{}".format(i))
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] projected = {}".format(projected))
      transformed.append(projected)

    # transformed = [c_0, h_0, a_0]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] transformed = {}".format(transformed))
    combined = transformed + decoder_state_flat[3:]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseDenseBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, combined)


class AttentionWrapperStatePairwiseWeightedSumBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """

    for s in [encoder_state, decoder_zero_state]:
      if not isinstance(s, tf.contrib.seq2seq.AttentionWrapperState):
        raise ValueError("AttentionWrapperStatePairwiseDenseBridge only supports linear transformation on AttentionWrapperState states")

    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] encoder_state = {}"
        .format(encoder_state))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] decoder_zero_state = {}"
      .format(decoder_zero_state))

    # Flattened states.
    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] encoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] decoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    # Extract decoder state size
      # decoder state sizes for h,c,a should be the same [batch, depth]
    decoder_state_size = decoder_state_flat[0].get_shape().as_list()[-1]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] decoder_state_size = {}"
        .format(decoder_state_size))

    # Apply linear transformation on [c_m, c_s], [h_m, h_s], [a_m, a_s] respectively
    batch_size = tf.shape(encoder_state_flat[0])[0]
    encoder_state_size = encoder_state_flat[0].get_shape().as_list()[-1]
    transformed = []
    for i in range(3):
      vec_in_m = tf.expand_dims(encoder_state_flat[i], axis=-1)
      vec_in_s = tf.expand_dims(decoder_state_flat[i], axis=-1)
      vec_in = tf.reshape(tf.concat([vec_in_m, vec_in_s], axis=-1), [-1, 2])
      tf.logging.info(" >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] vec_in = {}".format(vec_in))

      weights = tf.get_variable(
          name="bridge_weight_{}".format(i),
          shape=[2, 1],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer,
          trainable=True)
      weights = tf.nn.softmax(weights)
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] weights = {}".format(weights))

      projected = tf.squeeze(tf.reshape(tf.matmul(vec_in, weights), [batch_size, encoder_state_size, 1]), axis=-1)
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] projected = {}".format(projected))
      transformed.append(projected)

    # transformed = [c_0, h_0, a_0]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] transformed = {}".format(transformed))
    combined = transformed + decoder_state_flat[3:]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateWeightedSumBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, combined)

class AttentionWrapperStateAverageBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    for s in [encoder_state, decoder_zero_state]:
      if not isinstance(s, tf.contrib.seq2seq.AttentionWrapperState):
        raise ValueError(
          "AttentionWrapperStateAggregatedDenseBridge only supports linear transformation on AttentionWrapperState states")

    # Flattened states.
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] encoder_state = {}".format(encoder_state))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] decoder_zero_state = {}".format(decoder_zero_state))

    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(" >> [bridge.py class AttentionWrapperStateAverageBridge _build] encoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(" >> [bridge.py class AttentionWrapperStateAverageBridge _build] decoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    encoder_state_concat = tf.expand_dims(tf.concat(encoder_state_flat[:3], axis=1), axis=-1)  # (c_m, h_m, a_m)
    decoder_state_concat = tf.expand_dims(tf.concat(decoder_state_flat[:3], axis=1), axis=-1)  # (c_s, h_s, a_s)
    all_states = tf.concat([encoder_state_concat, decoder_state_concat], axis=-1) # b*d*2
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] encoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] decoder_state_concat = {}"
      .format(encoder_state_concat))

    # Extract decoder state sizes.
    encoder_state_size = [x.get_shape().as_list()[-1] for x in encoder_state_flat[:3]]
    encoder_total_size = sum(encoder_state_size)
    decoder_state_size = [x.get_shape().as_list()[-1] for x in decoder_state_flat[:3]]
    decoder_total_size = sum(decoder_state_size)
    assert encoder_total_size == decoder_total_size
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] encoder_total_size = decoder_state_size = {}"
        .format(decoder_state_size))

    transformed = tf.reduce_mean(all_states, axis=-1)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] transformed = {}".format(transformed))

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] splitted = {}".format(splitted))
    combined = splitted + decoder_state_flat[3:]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAverageBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, combined)

class AttentionWrapperStatePairwiseGatingBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """

    for s in [encoder_state, decoder_zero_state]:
      if not isinstance(s, tf.contrib.seq2seq.AttentionWrapperState):
        raise ValueError(
          "AttentionWrapperStateGatingBridge only supports linear transformation on AttentionWrapperState states")

    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] encoder_state = {}".format(encoder_state))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] decoder_zero_state = {}".format(decoder_zero_state))

    # Flattened states.
    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] encoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] decoder_state_flat = \n{}"
      .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    # Extract decoder state size
    # decoder state sizes for h,c,a should be the same [batch, depth]
    decoder_state_size = decoder_state_flat[0].get_shape().as_list()[-1]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] decoder_state_size = {}"
      .format(decoder_state_size))

    # Apply transformation on [c_m, c_s], [h_m, h_s], [a_m, a_s] respectively
    transformed = []
    for i in range(3):
      pairwise_gate_cell = tf.contrib.rnn.GRUCell(num_units=decoder_state_size, name='gate_cell_{}'.format(i))
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] pairwise_gate_cell.state_size = {}"
        .format(pairwise_gate_cell.state_size))

      vec_in_m = encoder_state_flat[i]
      vec_in_s = decoder_state_flat[i]
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] vec_in_m = {}".format(vec_in_m))
      tf.logging.info(
        " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] vec_in_s = {}".format(vec_in_s))

      '''
      Returns:
        A pair containing:
        Output: A 2-D tensor with shape [batch_size, self.output_size].
        New state: Either a single 2-D tensor, or a tuple of tensors matching the arity and shapes of state
      In this case, output == new_state
      '''

      new_state = pairwise_gate_cell(inputs=vec_in_s, state=vec_in_m)
      tf.logging.info(" >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] new_state = {}"
                      .format(new_state))

      transformed.append(new_state[0])

    # transformed = [c_0, h_0, a_0]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] transformed = {}".format(transformed))
    combined = transformed + decoder_state_flat[3:]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStatePairwiseGatingBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, combined)


class AttentionWrapperStateAggregatedGatingBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    for s in [encoder_state, decoder_zero_state]:
      if not isinstance(s, tf.contrib.seq2seq.AttentionWrapperState):
        raise ValueError(
          "AttentionWrapperStateAggregatedDenseBridge only supports linear transformation on AttentionWrapperState states")

    # Flattened states.
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] encoder_state = {}".format(encoder_state))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] decoder_zero_state = {}".format(decoder_zero_state))

    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(" >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] encoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in encoder_state_flat])))
    tf.logging.info(" >> [bridge.py class AttentionWrapperStateAggregatedDenseBridge _build] decoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in decoder_state_flat])))

    encoder_state_concat = tf.concat(encoder_state_flat[:3], 1)  # (c_m, h_m, a_m)
    decoder_state_concat = tf.concat(decoder_state_flat[:3], 1)  # (c_s, h_s, a_s)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] encoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] decoder_state_concat = {}"
      .format(encoder_state_concat))

    # Extract decoder state sizes.
    decoder_state_size = [x.get_shape().as_list()[-1] for x in decoder_state_flat[:3]]
    decoder_total_size = sum(decoder_state_size)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] decoder_state_size = {}"
        .format(decoder_state_size))

    # Apply GRU cell for gating
    aggregated_gate_cell = tf.contrib.rnn.GRUCell(num_units=decoder_total_size, name='gate_cell')
    transformed = aggregated_gate_cell(inputs=decoder_state_concat, state=encoder_state_concat)[0] # output
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] transformed = {}".format(transformed))

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] splitted = {}".format(splitted))
    combined = splitted + decoder_state_flat[3:]
    tf.logging.info(
      " >> [bridge.py class AttentionWrapperStateAggregatedGatingBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, combined)