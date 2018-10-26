"""Define bridges: logic of passing the encoder state to the decoder."""

import abc
import six

import tensorflow as tf
from opennmt.layers.bridge import Bridge

class NestedStateBridge(Bridge):
  """Base class for bridges."""

  def __call__(self, encoder_state, decoder_zero_state, sub_attention_over_encoder=False):


    '''
    NestedState(
      cell_state=LSTMStateTuple(
        c=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/lstm_cell/add_1:0' shape=(?, 128) dtype=float32>, 
        h=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/lstm_cell/mul_2:0' shape=(?, 128) dtype=float32>), 
      attention=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/concat_2:0' shape=(?, 128) dtype=float32>, 
      time=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/add:0' shape=() dtype=int32>, 
      alignments=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/Softmax:0' shape=(?, ?) dtype=float32>, alignment_history=<tensorflow.python.util.tf_should_use.TFShouldUseWarningWrapper object at 0x7f35d43e2bd0>, 
      attention_state=<tf.Tensor 'seq2seq/decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/Softmax:0' shape=(?, ?) dtype=float32>)
    
    By LuongAttention, attention_state = alignments


    "BeamSearchDecoderState": ("cell_state", "log_probs", "finished", "lengths")))
                              cell_state: LSTMStateTuple/AttentionWrapperState
    '''

    need_to_tile_master_state = False
    master_context_vector = None

    # ---------------------------- processing decoder_zero_state ---------------------------- #
    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] decoder_zero_state = {}".format(decoder_zero_state))
    # Flattened states.
    decoder_zero_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)
    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] decoder_zero_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in decoder_zero_state_flat])))

    if isinstance(decoder_zero_state, tf.contrib.seq2seq.AttentionWrapperState):
      decoder_state_num = 3
    elif isinstance(decoder_zero_state, tf.nn.rnn_cell.LSTMStateTuple):
      assert sub_attention_over_encoder is False
      decoder_state_num = 2
    elif isinstance(decoder_zero_state, tf.contrib.seq2seq.BeamSearchDecoderState):
      need_to_tile_master_state = True
      if isinstance(decoder_zero_state.cell_state, tf.nn.rnn_cell.LSTMStateTuple):
        assert sub_attention_over_encoder is False
        decoder_state_num = 2
      elif isinstance(decoder_zero_state.cell_state, tf.nn.rnn_cell.AttentionWrapperState):
        decoder_state_num = 3
      else:
        raise ValueError("what the _ is this !?! {}".format(type(decoder_zero_state.cell_state)))
      depth = decoder_zero_state.cell_state.h.get_shape().as_list()[-1]  # get static instead of dynamic shape
      beam_width = decoder_zero_state.cell_state.h.get_shape().as_list()[-2]
      tf.logging.info(" >> [bridge.py class NestedStateBridge _build] decoder_state_num = {}".format(decoder_state_num))
      tf.logging.info(" >> [bridge.py class NestedStateBridge _build] depth = {}".format(depth))
      tf.logging.info(" >> [bridge.py class NestedStateBridge _build] beam_width = {}".format(beam_width))

      decoder_zero_state_flat = [tf.reshape(s, [-1, depth]) for s in decoder_zero_state_flat[:decoder_state_num]] + decoder_zero_state_flat[decoder_state_num:]
    else:
      raise ValueError("NestedStateBridge only supports AttentionWrapperState, LSTMStateTuple "
                       "or BeamSearchDecoderState wrapping around these two")

    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] decoder_state_num = {}".format(decoder_state_num))

    # ---------------------------- processing encoder_state ---------------------------- #
    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] encoder_state = {}".format(encoder_state))
    # Flattened states.
    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] encoder_state_flat = \n{}"
                    .format("\n".join(["{}".format(x) for x in encoder_state_flat])))

    if isinstance(encoder_state, tf.contrib.seq2seq.AttentionWrapperState):
      encoder_state_num = 3
      master_context_vector = encoder_state_flat[2]
    elif isinstance(encoder_state, tf.nn.rnn_cell.LSTMStateTuple):
      assert sub_attention_over_encoder is False
      encoder_state_num = 2
    else:
      raise ValueError("NestedStateBridge currently only supports encoder_state: AttentionWrapperState, LSTMStateTuple")

    if need_to_tile_master_state:
      encoder_state_flat = [tf.contrib.seq2seq.tile_batch(s, multiplier=beam_width)
                                  for s in encoder_state_flat[:encoder_state_num]]

      tf.logging.info(" >> [bridge.py class NestedStateBridge _build] AFTER encoder_state_flat = \n{}"
                      .format("\n".join(["{}".format(x) for x in encoder_state_flat])))

    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] encoder_state_num = {}".format(encoder_state_num))

    state_tuples = (encoder_state_flat, encoder_state_num, decoder_zero_state_flat, decoder_state_num)

    bridge_output_flat = self._build(state_tuples)
    tf.logging.info(" >> [bridge.py class NestedStateBridge _build] bridge_output_flat = {}".format(bridge_output_flat))

    if need_to_tile_master_state:
      bridge_output_flat = [tf.reshape(s, [-1, beam_width, depth])
                            for s in bridge_output_flat[:decoder_state_num]] + bridge_output_flat[decoder_state_num:]

    bridge_output = tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, bridge_output_flat)
    return bridge_output, master_context_vector


class NestedStateAggregatedDenseBridge(NestedStateBridge):
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

  def _build(self, state_tuples):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    encoder_state_flat, encoder_state_num, decoder_state_flat, decoder_state_num = state_tuples

    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] encoder_state_num = {}"
        .format(encoder_state_num))
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] decoder_state_num = {}"
        .format(decoder_state_num))

    encoder_state_concat = tf.concat(encoder_state_flat[:encoder_state_num], 1) # (c_m, h_m, a_m)
    decoder_state_concat = tf.concat(decoder_state_flat[:decoder_state_num], 1) # (c_s, h_s, [a_s])
    aggregated_state_concat = tf.concat([encoder_state_concat, decoder_state_concat], 1)

    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] encoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] decoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] aggregated_state_concat = {}"
      .format(aggregated_state_concat))

    # Extract decoder state sizes.
    decoder_state_size = [x.get_shape().as_list()[-1] for x in decoder_state_flat[:decoder_state_num]]
    decoder_total_size = sum(decoder_state_size)
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] decoder_state_size = {}"
      .format(decoder_state_size))

    # Apply linear transformation.
    # TODO: maybe not using bias is better
    transformed = tf.layers.dense(
      aggregated_state_concat,
      decoder_total_size,
      activation=self.activation,
      name="nested_state_aggregated_dense_bridge_dense_layer")

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] splitted = {}".format(splitted))
    combined = splitted + decoder_state_flat[decoder_state_num:]
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedDenseBridge _build] combined = {}".format(combined))

    return combined

class NestedStatePairwiseDenseBridge(NestedStateBridge):
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

  def _build(self, state_tuples):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """

    encoder_state_flat, encoder_state_num, decoder_state_flat, decoder_state_num = state_tuples

    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseDenseBridge _build] encoder_state_num = {}"
        .format(encoder_state_num))
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseDenseBridge _build] decoder_state_num = {}"
        .format(decoder_state_num))

    # Extract decoder state size
      # decoder state sizes for h,c,a should be the same [batch, depth]
    decoder_state_size = decoder_state_flat[0].get_shape().as_list()[-1]
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseDenseBridge _build] decoder_state_size = {}"
      .format(decoder_state_size))

    # Apply linear transformation on [c_m, c_s], [h_m, h_s], [a_m, a_s] respectively
    # TODO: maybe not using bias is better
    transformed = []
    for i in range(decoder_state_num):
      vec_in = tf.concat([encoder_state_flat[i], decoder_state_flat[i]], 1)
      tf.logging.info(
        " >> [bridge.py class NestedStatePairwiseDenseBridge _build] vec_in = {}".format(vec_in))
      projected = tf.layers.dense(vec_in,
                                  decoder_state_size,
                                  activation=self.activation,
                                  name="nested_state_pairwise_dense_bridge_dense_layer_{}".format(i))
      tf.logging.info(
        " >> [bridge.py class NestedStatePairwiseDenseBridge _build] projected = {}".format(projected))
      transformed.append(projected)

    # transformed = [c_0, h_0, a_0]
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseDenseBridge _build] transformed = {}".format(transformed))
    combined = transformed + decoder_state_flat[decoder_state_num:]
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseDenseBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return combined

class NestedStatePairwiseWeightedSumBridge(NestedStateBridge):
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

  def _build(self, state_tuples):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """

    encoder_state_flat, encoder_state_num, decoder_state_flat, decoder_state_num = state_tuples

    tf.logging.info(
      " >> [bridge.py class NestedStateWeightedSumBridge _build] encoder_state_num = {}"
        .format(encoder_state_num))
    tf.logging.info(
      " >> [bridge.py class NestedStateWeightedSumBridge _build] decoder_state_num = {}"
        .format(decoder_state_num))

    # Extract decoder state size
      # decoder state sizes for h,c,a should be the same [batch, depth]
    decoder_state_size = decoder_state_flat[0].get_shape().as_list()[-1]
    tf.logging.info(
      " >> [bridge.py class NestedStateWeightedSumBridge _build] decoder_state_size = {}"
        .format(decoder_state_size))

    # Apply linear transformation on [c_m, c_s], [h_m, h_s], [a_m, a_s] respectively
    batch_size = tf.shape(encoder_state_flat[0])[0]
    encoder_state_size = encoder_state_flat[0].get_shape().as_list()[-1]
    transformed = []
    for i in range(decoder_state_num):
      vec_in_m = tf.expand_dims(encoder_state_flat[i], axis=-1)
      vec_in_s = tf.expand_dims(decoder_state_flat[i], axis=-1)
      vec_in = tf.reshape(tf.concat([vec_in_m, vec_in_s], axis=-1), [-1, 2])
      tf.logging.info(" >> [bridge.py class NestedStateWeightedSumBridge _build] vec_in = {}".format(vec_in))

      weights = tf.get_variable(
          name="nested_state_pairwise_weighted_sum_bridge_weight_{}".format(i),
          shape=[2, 1],
          dtype=tf.float32,
          initializer=tf.random_normal_initializer,
          trainable=True)
      weights = tf.nn.softmax(weights)
      tf.logging.info(
        " >> [bridge.py class NestedStateWeightedSumBridge _build] weights = {}".format(weights))

      projected = tf.squeeze(tf.reshape(tf.matmul(vec_in, weights), [batch_size, encoder_state_size, 1]), axis=-1)
      tf.logging.info(
        " >> [bridge.py class NestedStateWeightedSumBridge _build] projected = {}".format(projected))
      transformed.append(projected)

    # transformed = [c_0, h_0, a_0]
    tf.logging.info(
      " >> [bridge.py class NestedStateWeightedSumBridge _build] transformed = {}".format(transformed))
    combined = transformed + decoder_state_flat[decoder_state_num:]
    tf.logging.info(
      " >> [bridge.py class NestedStateWeightedSumBridge _build] combined = {}".format(combined))

    return combined

class NestedStateAverageBridge(NestedStateBridge):
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

  def _build(self, state_tuples):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_zero_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    encoder_state_flat, encoder_state_num, decoder_state_flat, decoder_state_num = state_tuples

    if encoder_state_num != decoder_state_num:
      raise ValueError("NestedStateAverageBridge only supports states of the same type !!! ")

    encoder_state_concat = tf.expand_dims(tf.concat(encoder_state_flat[:encoder_state_num], axis=1), axis=-1)  # (c_m, h_m, a_m)
    decoder_state_concat = tf.expand_dims(tf.concat(decoder_state_flat[:decoder_state_num], axis=1), axis=-1)  # (c_s, h_s, a_s)
    all_states = tf.concat([encoder_state_concat, decoder_state_concat], axis=-1) # b*d*2

    tf.logging.info(
      " >> [bridge.py class NestedStateAverageBridge _build] encoder_state_num = {}"
        .format(encoder_state_num))
    tf.logging.info(
      " >> [bridge.py class NestedStateAverageBridge _build] decoder_state_num = {}"
        .format(decoder_state_num))

    # Extract decoder state sizes.
    encoder_state_size = [x.get_shape().as_list()[-1] for x in encoder_state_flat[:encoder_state_num]]
    encoder_total_size = sum(encoder_state_size)
    decoder_state_size = [x.get_shape().as_list()[-1] for x in decoder_state_flat[:decoder_state_num]]
    decoder_total_size = sum(decoder_state_size)
    if encoder_total_size != decoder_total_size:
      raise ValueError("encoder_total_size must be equal to decoder_total_size but got {} and {}"
                       .format(encoder_total_size, decoder_total_size))
    tf.logging.info(
      " >> [bridge.py class NestedStateAverageBridge _build] encoder_total_size = decoder_state_size = {}"
        .format(decoder_state_size))

    transformed = tf.reduce_mean(all_states, axis=-1)
    tf.logging.info(
      " >> [bridge.py class NestedStateAverageBridge _build] transformed = {}".format(transformed))

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)
    tf.logging.info(
      " >> [bridge.py class NestedStateAverageBridge _build] splitted = {}".format(splitted))
    combined = splitted + decoder_state_flat[decoder_state_num:]
    tf.logging.info(
      " >> [bridge.py class NestedStateAverageBridge _build] combined = {}".format(combined))

    return combined

class NestedStatePairwiseGatingBridge(NestedStateBridge):
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

  def _build(self, state_tuples):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    encoder_state_flat, encoder_state_num, decoder_state_flat, decoder_state_num = state_tuples

    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseGatingBridge _build] encoder_state_num = {}"
        .format(encoder_state_num))
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseGatingBridge _build] decoder_state_num = {}"
        .format(decoder_state_num))

    # Extract decoder state size
    # decoder state sizes for h,c,a should be the same [batch, depth]
    decoder_state_size = decoder_state_flat[0].get_shape().as_list()[-1]
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseGatingBridge _build] decoder_state_size = {}"
      .format(decoder_state_size))

    # Apply transformation on [c_m, c_s], [h_m, h_s], [a_m, a_s] respectively
    transformed = []
    for i in range(decoder_state_num):
      pairwise_gate_cell = tf.contrib.rnn.GRUCell(num_units=decoder_state_size,
                                                  name='nested_state_pairwise_gating_bridge_gate_cell{}'.format(i))

      # pairwise_gate_cell = tf.contrib.rnn.GRUCell(num_units=decoder_state_size,
      #                                             name='gate_cell_{}'.format(i))

      tf.logging.info(
        " >> [bridge.py class NestedStatePairwiseGatingBridge _build] pairwise_gate_cell.state_size = {}"
        .format(pairwise_gate_cell.state_size))

      vec_in_m = encoder_state_flat[i]
      vec_in_s = decoder_state_flat[i]
      tf.logging.info(
        " >> [bridge.py class NestedStatePairwiseGatingBridge _build] vec_in_m = {}".format(vec_in_m))
      tf.logging.info(
        " >> [bridge.py class NestedStatePairwiseGatingBridge _build] vec_in_s = {}".format(vec_in_s))

      '''
      Returns:
        A pair containing:
        Output: A 2-D tensor with shape [batch_size, self.output_size].
        New state: Either a single 2-D tensor, or a tuple of tensors matching the arity and shapes of state
      In this case, output == new_state
      '''

      new_state = pairwise_gate_cell(inputs=vec_in_s, state=vec_in_m)
      tf.logging.info(" >> [bridge.py class NestedStatePairwiseGatingBridge _build] new_state = {}"
                      .format(new_state))

      transformed.append(new_state[0])

    # transformed = [c_0, h_0, a_0]
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseGatingBridge _build] transformed = {}".format(transformed))
    combined = transformed + decoder_state_flat[decoder_state_num:]
    tf.logging.info(
      " >> [bridge.py class NestedStatePairwiseGatingBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return combined

class NestedStateAggregatedGatingBridge(NestedStateBridge):
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

  def _build(self, state_tuples):
    """
    :param encoder_state: input state vectors for injecting information into this decoder
                          encoder_state = initial_state = next_state (master)

    :param decoder_state: original zero_state/previous_state that carries on the information
                               decoder_zero_state = previous_state = sub_state (sub)
    """
    encoder_state_flat, encoder_state_num, decoder_state_flat, decoder_state_num = state_tuples

    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] encoder_state_num = {}"
        .format(encoder_state_num))
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] decoder_state_num = {}"
        .format(decoder_state_num))

    encoder_state_concat = tf.concat(encoder_state_flat[:encoder_state_num], 1)  # (c_m, h_m, a_m)
    decoder_state_concat = tf.concat(decoder_state_flat[:decoder_state_num], 1)  # (c_s, h_s, a_s)
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] encoder_state_concat = {}"
      .format(encoder_state_concat))
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] decoder_state_concat = {}"
      .format(encoder_state_concat))

    # Extract decoder state sizes.
    decoder_state_size = [x.get_shape().as_list()[-1] for x in decoder_state_flat[:decoder_state_num]]
    decoder_total_size = sum(decoder_state_size)
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] decoder_state_size = {}"
        .format(decoder_state_size))

    # Apply GRU cell for gating
    aggregated_gate_cell = tf.contrib.rnn.GRUCell(num_units=decoder_total_size, name='nested_state_aggregated_gating_bridge_gate_cell')
    transformed = aggregated_gate_cell(inputs=decoder_state_concat, state=encoder_state_concat)[0] # output
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] transformed = {}".format(transformed))

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] splitted = {}".format(splitted))
    combined = splitted + decoder_state_flat[decoder_state_num:]
    tf.logging.info(
      " >> [bridge.py class NestedStateAggregatedGatingBridge _build] combined = {}".format(combined))

    # Pack as the original decoder state.
    return combined
