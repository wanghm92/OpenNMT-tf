# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Seq2seq layer operations for use in neural networks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

import tensorflow as tf

from opennmt.layers.reducer import align_in_time_transposed_nest

__all__ = ["TfContribSeq2seqDecoder", "hierarchical_dynamic_decode"]

_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

def _transpose_back_batch_mastertime_subtime(x):
  """Transposes the batch and time dimensions of a Tensor.

  If the input tensor has rank < 2 it returns the original tensor. Retains as
  much of the static shape information as possible.

  Args:
    x: A Tensor.

  Returns:
    x transposed along the 1st and 2nd dimensions. (0th dimension unchanged)
  """
  x_static_shape = x.get_shape()
  if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
    return x

  x_rank = array_ops.rank(x)
  x_t = array_ops.transpose(
      x, array_ops.concat(
          ([2, 0, 1], math_ops.range(3, x_rank)), axis=0))
  x_t.set_shape(
      tensor_shape.TensorShape([
          x_static_shape[2].value, x_static_shape[0].value, x_static_shape[1].value
      ]).concatenate(x_static_shape[3:]))
  return x_t

def _unstack_ta(inp):
  return tensor_array_ops.TensorArray(
      dtype=inp.dtype, size=array_ops.shape(inp)[0],
      element_shape=inp.get_shape()[1:]).unstack(inp)


# ------------------------------------------------------------------------------------------------------------------ #
# ------------------------------------- Class HierarchicalDynamicDecoder ------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

@six.add_metaclass(abc.ABCMeta)
class TfContribSeq2seqDecoder(object):
  """An RNN Decoder abstract interface object.

  Overwritting tf.contib.seq2seq.ops.decoder.Decoder, just not to confuse with opennmt.Decoder class

  Concepts used by this interface:
  - `inputs`: (structure of) tensors and TensorArrays that is passed as input to
    the RNNCell composing the decoder, at each time step.
  - `state`: (structure of) tensors and TensorArrays that is passed to the
    RNNCell instance as the state.
  - `finished`: boolean tensor telling whether each sequence in the batch is
    finished.
  - `outputs`: Instance of BasicDecoderOutput. Result of the decoding, at each
    time step.
  """

  @property
  def batch_size(self):
    """The batch size of input values."""
    raise NotImplementedError

  @property
  def output_size(self):
    """A (possibly nested tuple of...) integer[s] or `TensorShape` object[s]."""
    raise NotImplementedError

  @property
  def output_dtype(self):
    """A (possibly nested tuple of...) dtype[s]."""
    raise NotImplementedError

  @abc.abstractmethod
  def initialize(self, name=None):
    """Called before any decoding iterations.

    This methods must compute initial input values and initial state.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, initial_inputs, initial_state)`: initial values of
      'finished' flags, inputs and state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def step(self, time, inputs, state, name=None):
    """Called per step of decoding (but only once for dynamic decoding).

    Args:
      time: Scalar `int32` tensor. Current step number.
      inputs: RNNCell input (possibly nested tuple of) tensor[s] for this time
        step.
      state: RNNCell state (possibly nested tuple of) tensor[s] from previous
        time step.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`: `outputs` is an object
      containing the decoder output, `next_state` is a (structure of) state
      tensors and TensorArrays, `next_inputs` is the tensor that should be used
      as input for the next step, `finished` is a boolean tensor telling whether
      the sequence is complete, for each sequence in the batch.
    """
    raise NotImplementedError

  def finalize(self, outputs, final_state, sequence_lengths):
    raise NotImplementedError

  @property
  def tracks_own_finished(self):
    """Describes whether the Decoder keeps track of finished states.

    Most decoders will emit a true/false `finished` value independently
    at each time step.  In this case, the `dynamic_decode` function keeps track
    of which batch entries are already finished, and performs a logical OR to
    insert new batches to the finished set.

    Some decoders, however, shuffle batches / beams between time steps and
    `dynamic_decode` will mix up the finished state across these entries because
    it does not track the reshuffle across time steps.  In this case, it is
    up to the decoder to declare that it will keep track of its own finished
    state by setting this property to `True`.

    Returns:
      Python bool.
    """
    return False

# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- def hierarchical_dynamic_decode() ---------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

def _create_zero_outputs(size, dtype, batch_size):
  """Create a zero outputs Tensor structure."""
  def _create(s, d):
    return _zero_state_tensors(s, batch_size, d)

  return nest.map_structure(_create, size, dtype)

def hierarchical_dynamic_decode(
        master_decoder,
        sub_decoder,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=None,
        sub_maximum_iterations=None,
        parallel_iterations=32,
        swap_memory=False,
        scope=None,
        dynamic=False,
        shifted=None,
        pass_master_state=False,
        sub_attention_over_encoder=False):
  """Perform dynamic decoding with `master_decoder`.

  Calls initialize() once and step() repeatedly on the Decoder object.

  Args:
    master_decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the master_decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.

  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.

  Raises:
    TypeError: if `master_decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
  if not isinstance(master_decoder, TfContribSeq2seqDecoder):
    raise TypeError("Expected master_decoder to be type Decoder, but saw: %s" %
                    type(master_decoder))
  with variable_scope.variable_scope(scope, "decoder") as varscope:
    # Determine context types.
    ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
    is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
    in_while_loop = (
        control_flow_util.GetContainingWhileContext(ctxt) is not None)
    # Properly cache variable values inside the while_loop.
    # Don't set a caching device when running in a loop, since it is possible
    # that train steps could be wrapped in a tf.while_loop. In that scenario
    # caching prevents forward computations in loop iterations from re-reading
    # the updated weights.
    if not context.executing_eagerly() and not in_while_loop:
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
      if maximum_iterations.get_shape().ndims != 0:
        raise ValueError("maximum_iterations must be a scalar")

    initial_finished, initial_inputs, initial_state = master_decoder.initialize()
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_finished "
                    "(finished = math_ops.equal(0, self._sequence_length)) = {}".format(initial_finished))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_inputs "
                    "(self._input_tas.read(0)) = {}".format(initial_inputs))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_state = {}".format(initial_state))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] shifted = {}".format(shifted))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] pass_master_state = {}".format(pass_master_state))

    zero_outputs = _create_zero_outputs(master_decoder.output_size,
                                        master_decoder.output_dtype,
                                        master_decoder.batch_size)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] zero_outputs = {}".format(zero_outputs))

    if is_xla and maximum_iterations is None:
      raise ValueError("maximum_iterations is required for XLA compilation.")
    if maximum_iterations is not None:
      initial_finished = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)
    initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
    initial_time = constant_op.constant(0, dtype=dtypes.int32)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_sequence_lengths = {}".format(initial_sequence_lengths))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_time = {}".format(initial_time))

    def _shape(batch_size, from_shape):
      if not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0:
        return tensor_shape.TensorShape(None)
      else:
        batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name="batch_size"))
        return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    dynamic_size = maximum_iterations is None or not is_xla
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] dynamic_size = {}".format(dynamic_size))

    def _create_ta(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=_shape(master_decoder.batch_size, s))

    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] master_decoder.output_size = {}".format(master_decoder.output_size))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] master_decoder.batch_size = {}".format(master_decoder.batch_size))
    initial_outputs_ta = nest.map_structure(_create_ta, master_decoder.output_size, master_decoder.output_dtype)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_outputs_ta = {}".format(initial_outputs_ta))

    def _create_ta_sub(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=tensor_shape.TensorShape(None) if dynamic else _shape(sub_decoder.sub_time, _shape(sub_decoder.batch_size, s)),  # sub_decoder.sub_time is dynamic shape
          infer_shape=False)

    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] sub_decoder.output_size = {}".format(sub_decoder.output_size))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] sub_decoder.batch_size = {}".format(sub_decoder.batch_size))
    initial_outputs_ta_sub = nest.map_structure(_create_ta_sub, sub_decoder.output_size, sub_decoder.output_dtype)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_outputs_ta_sub = {}".format(initial_outputs_ta_sub))

    def _create_ta_general(d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=tensor_shape.TensorShape(None),
          infer_shape=False)

    # store the sequence_length ([batch]) tensors for sub-decoder
    initial_sequence_mask_ta_sub = nest.map_structure(_create_ta_general, tf.float32)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] initial_sequence_lengths_ta_sub = {}".format(initial_sequence_mask_ta_sub))

    sub_maximum_iterations = sub_maximum_iterations if dynamic else sub_decoder.sub_time
    initial_sub_state = sub_decoder._initial_state

    def condition(unused_time, unused_outputs_ta, unused_outputs_ta_sub, unused_state, unused_inputs,
                  finished, unused_sequence_lengths, unused_sequence_lengths_ta_sub, unused_next_sub_state, unused_previous_sub_inputs):
      return math_ops.logical_not(math_ops.reduce_all(finished))

    def body(time, outputs_ta, outputs_ta_sub, state, inputs, finished, sequence_lengths, sequence_mask_ta_sub, previous_sub_state, previous_sub_inputs):
      """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
      next_outputs, decoder_state, next_inputs, decoder_finished = master_decoder.step(time, inputs, state)

      if master_decoder.tracks_own_finished:
        next_finished = decoder_finished
      else:
        next_finished = math_ops.logical_or(decoder_finished, finished)
      next_sequence_lengths = array_ops.where(
          math_ops.logical_not(finished),
          array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
          sequence_lengths)

      nest.assert_same_structure(state, decoder_state)
      nest.assert_same_structure(outputs_ta, next_outputs)
      nest.assert_same_structure(inputs, next_inputs)

      # Zero out output values past finish
      if impute_finished:
        emit = nest.map_structure(
            lambda out, zero: array_ops.where(finished, zero, out),
            next_outputs,
            zero_outputs)
      else:
        emit = next_outputs

      # Copy through states past finish
      def _maybe_copy_state(new, cur):
        # TensorArrays and scalar states get passed through.
        if isinstance(cur, tensor_array_ops.TensorArray):
          pass_through = True
        else:
          new.set_shape(cur.shape)
          pass_through = (new.shape.ndims == 0)
        return new if pass_through else array_ops.where(finished, cur, new)

      if impute_finished:
        next_state = nest.map_structure(_maybe_copy_state, decoder_state, state)
      else:
        next_state = decoder_state

      outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, emit)

      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] next_outputs = {}".format(next_outputs))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] next_state = {}".format(next_state))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] next_inputs = {}".format(next_inputs))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] next_finished = {}".format(next_finished))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] next_sequence_lengths = {}".format(next_sequence_lengths))

      """
        Beginning sub-decoding
      """
      if shifted == "attr":
          previous_inputs = next_inputs
      elif shifted == "word":
          previous_inputs = previous_sub_inputs
      else:
          previous_inputs = None

      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] previous_inputs = {}".format(previous_inputs))

      sub_outputs, next_sub_state, sub_length, sub_final_time, next_sub_inputs = sub_dynamic_decode(sub_decoder,
                                                                                                    master_time=time,
                                                                                                    master_input=next_inputs,
                                                                                                    master_state=next_state if pass_master_state else None,
                                                                                                    previous_state=previous_sub_state,
                                                                                                    previous_inputs=previous_inputs,
                                                                                                    maximum_iterations=sub_maximum_iterations,
                                                                                                    dynamic=dynamic,
                                                                                                    shifted=shifted,
                                                                                                    sub_attention_over_encoder=sub_attention_over_encoder)

      if not dynamic:
          sub_outputs = align_in_time_transposed_nest(sub_outputs, sub_decoder.sub_time)

      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] sub_outputs = {}".format(sub_outputs))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] next_sub_state = {}".format(next_sub_state))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] previous_sub_state = {}".format(previous_sub_state))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] sub_length = {}".format(sub_length))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] sub_final_time = {}".format(sub_final_time))

      # should not mask before log_softmax, should store the sequence mask instead
      sub_sequence_mask = nest.map_structure(lambda length: tf.sequence_mask(length, maxlen=sub_decoder.sub_time, dtype=tf.float32), sub_length)

      # zero length for sub sequence if master finished during dynamic_decoding
      if dynamic:
          master_mask = tf.expand_dims(tf.cast(math_ops.logical_not(next_finished), dtype=tf.float32), axis=-1)
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] master_mask = {}".format(master_mask))
          sub_sequence_mask = sub_sequence_mask * master_mask
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] sub_sequence_mask = {}".format(sub_sequence_mask))

      sub_sequence_mask = nest.map_structure(_transpose_batch_time, sub_sequence_mask) #[st, batch]

      outputs_ta_sub = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta_sub, sub_outputs)
      sequence_mask_ta_sub = nest.map_structure(lambda ta, out: ta.write(time, out), sequence_mask_ta_sub, sub_sequence_mask)

      return time + 1, outputs_ta, outputs_ta_sub, next_state, next_inputs, next_finished, next_sequence_lengths, sequence_mask_ta_sub, next_sub_state, next_sub_inputs

    res = control_flow_ops.while_loop(
        condition,
        body,
        loop_vars=(
            initial_time,
            initial_outputs_ta,
            initial_outputs_ta_sub,
            initial_state,
            initial_inputs,
            initial_finished,
            initial_sequence_lengths,
            initial_sequence_mask_ta_sub,
            initial_sub_state,
            initial_inputs,
        ),
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        swap_memory=swap_memory)

    final_time = res[0]
    final_outputs_ta = res[1]
    final_outputs_ta_sub = res[2]
    final_state = res[3]
    final_sequence_lengths = res[6]
    final_sequence_mask_ta_sub = res[7]
    final_state_sub = res[8]

    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] final_outputs_ta = {}".format(final_outputs_ta))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] final_state = {}".format(final_state))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] final_state_sub = {}".format(final_state_sub))

    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    def _aggregate_fn(ta):
        return ta.concat() if dynamic else ta.stack()
    final_outputs_sub = nest.map_structure(_aggregate_fn, final_outputs_ta_sub)  # [sum_of_st, batch, depth]
    final_sequence_mask_sub = nest.map_structure(_aggregate_fn, final_sequence_mask_ta_sub)  # [sum_of_st, batch]

    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] final_outputs = {}".format(final_outputs))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] dynamic = {}\nfinal_outputs_sub = {} ".format(dynamic, final_outputs_sub))

    try:
      final_outputs, final_state = master_decoder.finalize(final_outputs, final_state, final_sequence_lengths, final_sequence_mask_sub)
    except NotImplementedError:
      pass

    if not output_time_major:
        tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py hierarchical_dynamic_decode] output_time_major = {}".format(output_time_major))
        final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)

        _transpose_fn = _transpose_batch_time if dynamic else _transpose_back_batch_mastertime_subtime
        final_outputs_sub = nest.map_structure(_transpose_fn, final_outputs_sub) # [batch, mt, st, depth] / [batch, sum_of_st, depth]
        final_sequence_mask_sub = nest.map_structure(_transpose_fn, final_sequence_mask_sub)  # [batch, mt, st] / [batch, sum_of_st]

  return final_outputs, final_outputs_sub, final_state, final_state_sub, final_sequence_lengths, final_sequence_mask_sub, final_time


# --------------------------------------------------------------------------------------------------------- #
# --------------------------------------- def sub_dynamic_decode() ---------------------------------------- #
# --------------------------------------------------------------------------------------------------------- #

def sub_dynamic_decode(
        decoder,
        master_time,
        master_input=None,
        master_state=None,
        previous_state=None,
        previous_inputs=None,
        impute_finished=True,
        maximum_iterations=None,
        parallel_iterations=32,
        swap_memory=False,
        scope=None,
        dynamic=False,
        shifted=None,
        sub_attention_over_encoder=False):
  """Perform dynamic decoding with `decoder`.

  Calls initialize() once and step() repeatedly on the Decoder object.

  Args:
    decoder: A `Decoder` instance.
    output_time_major: Python boolean.  Default: `False` (batch major).  If
      `True`, outputs are returned as time major tensors (this mode is faster).
      Otherwise, outputs are returned as batch major tensors (this adds extra
      time to the computation).
    impute_finished: Python boolean.  If `True`, then states for batch
      entries which are marked as finished get copied through and the
      corresponding outputs get zeroed out.  This causes some slowdown at
      each time step, but ensures that the final state and outputs have
      the correct values and that backprop ignores time steps that were
      marked as finished.
    maximum_iterations: `int32` scalar, maximum allowed number of decoding
       steps.  Default is `None` (decode until the decoder is fully done).
    parallel_iterations: Argument passed to `tf.while_loop`.
    swap_memory: Argument passed to `tf.while_loop`.
    scope: Optional variable scope to use.

  Returns:
    `(final_outputs, final_state, final_sequence_lengths)`.

  Raises:
    TypeError: if `decoder` is not an instance of `Decoder`.
    ValueError: if `maximum_iterations` is provided but is not a scalar.
  """
  if not isinstance(decoder, TfContribSeq2seqDecoder):
    raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                    type(decoder))

  with variable_scope.variable_scope(scope, "sub_decoder") as varscope:
    # Determine context types.
    ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
    is_xla = control_flow_util.GetContainingXLAContext(ctxt) is not None
    in_while_loop = (
        control_flow_util.GetContainingWhileContext(ctxt) is not None)
    # Properly cache variable values inside the while_loop.
    # Don't set a caching device when running in a loop, since it is possible
    # that train steps could be wrapped in a tf.while_loop. In that scenario
    # caching prevents forward computations in loop iterations from re-reading
    # the updated weights.
    if not context.executing_eagerly() and not in_while_loop:
      if varscope.caching_device is None:
        varscope.set_caching_device(lambda op: op.device)

    if maximum_iterations is not None:
      maximum_iterations = ops.convert_to_tensor(
          maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")
      if maximum_iterations.get_shape().ndims != 0:
        raise ValueError("maximum_iterations must be a scalar")

    '''
        initial_state=next_state,
        previous_state=sub_state,
    '''
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] shifted = {}".format(shifted))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] master_input = {}".format(master_input))

    initial_finished, initial_inputs, initial_state = decoder.initialize(master_state=master_state,
                                                                         previous_state=previous_state,
                                                                         master_time=master_time,
                                                                         sub_attention_over_encoder=sub_attention_over_encoder)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] initial_finished = {}".format(initial_finished))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] decoder.batch_size = {}".format(decoder.batch_size))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] maximum_iterations = {}".format(maximum_iterations))
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] is_xla = {}".format(is_xla))

    if dynamic and shifted is not None and previous_inputs is not None:
        initial_inputs = previous_inputs
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] initial_inputs = {}".format(initial_inputs))

    zero_outputs = _create_zero_outputs(decoder.output_size,
                                        decoder.output_dtype,
                                        decoder.batch_size)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] zero_outputs = {}".format(zero_outputs))

    if is_xla and maximum_iterations is None:
      raise ValueError("maximum_iterations is required for XLA compilation.")
    if maximum_iterations is not None:
      initial_finished = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)
    initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
    initial_time = constant_op.constant(0, dtype=dtypes.int32)

    def _shape(batch_size, from_shape):
      if not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0:
        return tensor_shape.TensorShape(None)
      else:
        batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name="batch_size"))
        return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    dynamic_size = maximum_iterations is None or not is_xla
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] dynamic_size = {}".format(dynamic_size))

    def _create_ta(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=_shape(decoder.batch_size, s))

    initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)
    tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] initial_outputs_ta = {}".format(initial_outputs_ta))

    def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs, finished, unused_sequence_lengths, unused_final_inputs):
        return math_ops.logical_not(math_ops.reduce_all(finished))

    def body(time, outputs_ta, state, inputs, finished, sequence_lengths, final_inputs):
      """Internal while_loop body.

      Args:
        time: scalar int32 tensor.
        outputs_ta: structure of TensorArray.
        state: (structure of) state tensors and TensorArrays.
        inputs: (structure of) input tensors.
        finished: bool tensor (keeping track of what's finished).
        sequence_lengths: int32 tensor (keeping track of time of finish).

      Returns:
        `(time + 1, outputs_ta, next_state, next_inputs, next_finished,
          next_sequence_lengths)`.
        ```
      """
      # concatenate master input to word embeddings here
      if master_input is not None:
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] master_input = {}".format(master_input))
          master_emb_weight = tf.constant(0.5, dtype=inputs.dtype)
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] master_emb_weight = {}".format(master_emb_weight))
          master_input_weighted = tf.multiply(master_input, master_emb_weight)
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] master_input_weighted = {}".format(master_input_weighted))
          inputs_concat = tf.concat([inputs, master_input_weighted], -1)
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] inputs_concat = {}".format(inputs_concat))
          inputs_norm = tf.sqrt(tf.reduce_sum(tf.square(inputs_concat), axis=-1, keepdims=True))
          tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] inputs_norm = {}".format(inputs_norm))
          rnn_inputs = inputs_concat / inputs_norm
      else:
          rnn_inputs = inputs
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] rnn_inputs = {}".format(rnn_inputs))

      (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(time, rnn_inputs, state)
      if decoder.tracks_own_finished:
        next_finished = decoder_finished
      else:
        next_finished = math_ops.logical_or(decoder_finished, finished)
      next_sequence_lengths = array_ops.where(
          math_ops.logical_not(finished),
          array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
          sequence_lengths)
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] next_outputs = {}".format(next_outputs))
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] next_finished = {}".format(next_finished))

      nest.assert_same_structure(state, decoder_state)
      nest.assert_same_structure(outputs_ta, next_outputs)
      nest.assert_same_structure(inputs, next_inputs)

      # Zero out output values past finish
      if impute_finished:
        emit = nest.map_structure(
            lambda out, zero: array_ops.where(finished, zero, out),
            next_outputs,
            zero_outputs)
      else:
        emit = next_outputs
      tf.logging.info(" >> [tf_contrib_seq2seq_decoder.py sub_dynamic_decode] emit = {}".format(emit))

      # Copy through states past finish
      def _maybe_copy_state(new, cur):
        # TensorArrays and scalar states get passed through.
        if isinstance(cur, tensor_array_ops.TensorArray):
          pass_through = True
        else:
          new.set_shape(cur.shape)
          pass_through = (new.shape.ndims == 0)
        return new if pass_through else array_ops.where(finished, cur, new)

      if impute_finished:
        next_state = nest.map_structure(_maybe_copy_state, decoder_state, state)
      else:
        next_state = decoder_state

      outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, emit)

      if dynamic and shifted is not None:
          final_inputs = nest.map_structure(
              lambda out, zero: array_ops.where(next_finished, out, zero),
              inputs,
              final_inputs)

      return (time + 1, outputs_ta, next_state, next_inputs, next_finished, next_sequence_lengths, final_inputs)

    res = control_flow_ops.while_loop(
        condition,
        body,
        loop_vars=(
            initial_time,
            initial_outputs_ta,
            initial_state,
            initial_inputs,
            initial_finished,
            initial_sequence_lengths,
            initial_inputs,
        ),
        parallel_iterations=parallel_iterations,
        maximum_iterations=maximum_iterations,
        swap_memory=swap_memory)

    final_time = res[0]
    final_outputs_ta = res[1]
    final_state = res[2]
    final_sequence_lengths = res[5]
    final_inputs = res[6]

    final_outputs = tf.cond(
        math_ops.reduce_all(initial_finished),
        true_fn=lambda: nest.map_structure(lambda t: tf.expand_dims(t, axis=0), zero_outputs),
        false_fn=lambda: nest.map_structure(lambda ta: ta.stack(), final_outputs_ta))

    try:
      final_outputs, final_state = decoder.finalize(final_outputs, final_state, final_sequence_lengths)
    except NotImplementedError:
      pass

  return final_outputs, final_state, final_sequence_lengths, final_time, final_inputs