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

from opennmt.decoders.tf_contrib_seq2seq_decoder import _transpose_back_batch_mastertime_subtime, _create_zero_outputs, TfContribSeq2seqDecoder

__all__ = ["hierarchical_dynamic_decode_and_search"]

_transpose_batch_time = rnn._transpose_batch_time  # pylint: disable=protected-access
_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access

# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- def hierarchical_dynamic_decode_and_search() ----------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

def hierarchical_dynamic_decode_and_search(
        master_decoder,
        sub_decoder,
        sub_emb_gate=None,
        sub_bridge=None,
        beam_width=5,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=None,
        sub_maximum_iterations=None,
        parallel_iterations=32,
        swap_memory=False,
        scope=None,
        dynamic=True,
        force_non_rep=True,
        shifted=None,
        pass_master_state=False,
        pass_master_input=False,
        master_attention_at_input=False,
        htm1_at_emb_gate=False,
        ):
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
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_finished "
                    "(finished = math_ops.equal(0, self._sequence_length)) = {}".format(initial_finished))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_inputs "
                    "(self._input_tas.read(0)) = {}".format(initial_inputs))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_state = {}".format(initial_state))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] shifted = {}".format(shifted))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] pass_master_state = {}".format(pass_master_state))

    zero_outputs = _create_zero_outputs(master_decoder.output_size,
                                        master_decoder.output_dtype,
                                        master_decoder.batch_size)
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] zero_outputs = {}".format(zero_outputs))

    if is_xla and maximum_iterations is None:
      raise ValueError("maximum_iterations is required for XLA compilation.")
    if maximum_iterations is not None:
      initial_finished = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)

    initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
    initial_time = constant_op.constant(0, dtype=dtypes.int32)
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_sequence_lengths = {}".format(initial_sequence_lengths))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_time = {}".format(initial_time))

    def _shape(batch_size, from_shape):
      if not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0:
        return tensor_shape.TensorShape(None)
      else:
        batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name="batch_size"))
        return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

    dynamic_size = maximum_iterations is None or not is_xla
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] dynamic_size = {}".format(dynamic_size))

    def _create_ta(s, d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=_shape(master_decoder.batch_size, s))

    initial_outputs_ta = nest.map_structure(_create_ta, master_decoder.output_size, master_decoder.output_dtype)
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_outputs_ta = {}".format(initial_outputs_ta))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] master_decoder.output_size = {}".format(master_decoder.output_size))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] master_decoder.batch_size = {}".format(master_decoder.batch_size))

    # ---------------------------------------- preparing for sub decoding ---------------------------------------- #
    _, initial_inputs_sub, initial_state_sub = sub_decoder.initialize()
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_inputs_sub = {}".format(initial_inputs_sub))

    initial_outputs_ta_sub_dtype = tf.contrib.seq2seq.FinalBeamSearchDecoderOutput(beam_search_decoder_output=sub_decoder.output_dtype,
                                                                                   predicted_ids=sub_decoder.output_dtype.predicted_ids)
    def _create_ta_general(d):
      return tensor_array_ops.TensorArray(
          dtype=d,
          size=0 if dynamic_size else maximum_iterations,
          dynamic_size=dynamic_size,
          element_shape=tensor_shape.TensorShape(None),
          infer_shape=False)

    initial_outputs_ta_sub = nest.map_structure(_create_ta_general, initial_outputs_ta_sub_dtype)
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_outputs_ta_sub = {}".format(initial_outputs_ta_sub))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sub_decoder.output_dtype = {}".format(sub_decoder.output_dtype))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sub_decoder.batch_size = {}".format(sub_decoder.batch_size))

    # store the sequence_length ([batch]) tensors for sub-decoder
    initial_sequence_mask_ta_sub = nest.map_structure(_create_ta_general, tf.float32)
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] initial_sequence_lengths_ta_sub = {}".format(initial_sequence_mask_ta_sub))

    sub_maximum_iterations = sub_maximum_iterations
    # ---------------------------------------- Done preparing for sub decoding ---------------------------------------- #

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
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_finished = {}".format(next_finished))

      next_sequence_lengths = array_ops.where(
          math_ops.logical_not(finished),
          array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
          sequence_lengths)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_sequence_lengths = {}".format(next_sequence_lengths))

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] state = {}".format(state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] decoder_state = {}".format(decoder_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] outputs_ta = {}".format(outputs_ta))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_outputs = {}".format(next_outputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] inputs = {}".format(inputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_inputs = {}".format(next_inputs))

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

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] emit = {}".format(emit))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_state = {}".format(next_state))

      # ---------------------------------------------------------------------------------------------------------- #
      # --------------------------------------- Begin sub-decoding ----------------------------------------------- #
      # ---------------------------------------------------------------------------------------------------------- #

      previous_inputs = next_inputs if shifted == "attr" else previous_sub_inputs if shifted == "word" else None
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] previous_sub_inputs = {}".format(previous_sub_inputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_inputs = {}".format(next_inputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] previous_inputs = {}".format(previous_inputs))

      sub_outputs, next_sub_state, sub_length, sub_final_time, next_sub_inputs = sub_dynamic_decode_and_search(
                                                                                                    sub_decoder,
                                                                                                    sub_emb_gate=sub_emb_gate,
                                                                                                    sub_bridge=sub_bridge,
                                                                                                    beam_width=beam_width,
                                                                                                    master_input=next_inputs if pass_master_input else None,
                                                                                                    master_state=next_state if pass_master_state else None,
                                                                                                    previous_state=previous_sub_state,
                                                                                                    previous_inputs=previous_inputs,
                                                                                                    maximum_iterations=sub_maximum_iterations,
                                                                                                    force_non_rep=force_non_rep,
                                                                                                    shifted=shifted,
                                                                                                    master_attention_at_input=master_attention_at_input,
                                                                                                    htm1_at_emb_gate=htm1_at_emb_gate)

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_sub_inputs = {}".format(next_sub_inputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] next_sub_state = {}".format(next_sub_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] previous_sub_state = {}".format(previous_sub_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sub_length = {}".format(sub_length))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sub_final_time = {}".format(sub_final_time))

      # should not mask before log_softmax, should store the sequence mask instead
      sub_sequence_mask = nest.map_structure(lambda length: tf.sequence_mask(length, maxlen=None, dtype=tf.float32), sub_length)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] BEFORE sub_sequence_mask = {}".format(sub_sequence_mask))

      # zero length for sub sequence if master finished during dynamic_decoding
      master_mask = tf.expand_dims(tf.expand_dims(tf.cast(math_ops.logical_not(next_finished), dtype=tf.float32), axis=-1), axis=-1)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] master_mask = {}".format(master_mask))
      sub_sequence_mask = sub_sequence_mask * master_mask
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] AFTER sub_sequence_mask = {}".format(sub_sequence_mask))

      sub_sequence_mask = nest.map_structure(_transpose_back_batch_mastertime_subtime, sub_sequence_mask) #[st, batch, beam_size]

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] outputs_ta_sub = {}".format(outputs_ta_sub))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sub_outputs = {}".format(sub_outputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sequence_mask_ta_sub = {}".format(sequence_mask_ta_sub))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] AFTER_2 sub_sequence_mask = {}".format(sub_sequence_mask))

      outputs_ta_sub = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta_sub, sub_outputs)
      sequence_mask_ta_sub = nest.map_structure(lambda ta, out: ta.write(time, out), sequence_mask_ta_sub, sub_sequence_mask)

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] outputs_ta_sub = {}".format(outputs_ta_sub))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] sequence_mask_ta_sub = {}".format(sequence_mask_ta_sub))

      # -------------------------------------------------------------------------------------------------------- #
      # --------------------------------------- End sub-decoding ----------------------------------------------- #
      # -------------------------------------------------------------------------------------------------------- #

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
            initial_state_sub,
            initial_inputs_sub,
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

    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] final_outputs_ta = {}".format(final_outputs_ta))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] final_state = {}".format(final_state))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] final_state_sub = {}".format(final_state_sub))

    final_outputs = nest.map_structure(lambda ta: ta.stack(), final_outputs_ta)

    final_outputs_sub = nest.map_structure(lambda ta: ta.concat(), final_outputs_ta_sub)  # [sum_of_st, batch, beam_size]
    final_sequence_mask_sub = nest.map_structure(lambda ta: ta.concat(), final_sequence_mask_ta_sub)  # [sum_of_st, batch, beam_size]

    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] final_outputs = {}".format(final_outputs))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] final_outputs_sub = {} ".format(final_outputs_sub))

    try:
      final_outputs, final_state = master_decoder.finalize(final_outputs, final_state, final_sequence_lengths, final_sequence_mask_sub)
    except NotImplementedError:
      pass

    if not output_time_major:
        tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] output_time_major = {}".format(output_time_major))
        final_outputs = nest.map_structure(_transpose_batch_time, final_outputs)
        final_outputs_sub = nest.map_structure(_transpose_batch_time, final_outputs_sub) # [batch, sum_of_st, beam_size]
        final_sequence_mask_sub = nest.map_structure(_transpose_batch_time, final_sequence_mask_sub)  # [batch, sum_of_st, beam_size]

    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] AFTER final_outputs = {}".format(final_outputs))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] AFTER final_outputs_sub = {}".format(final_outputs_sub))
    tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py hierarchical_dynamic_decode_and_search] AFTER final_sequence_mask_sub = {}".format(final_sequence_mask_sub))

  return final_outputs, final_outputs_sub, final_state, final_state_sub, final_sequence_lengths, final_sequence_mask_sub, final_time


# -------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- def sub_dynamic_decode_and_search() ---------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #

def _init_sub_state(zero_state, sub_bridge=None, previous_state=None, master_state=None):

    # reset "log_probs", "finished", "lengths" in previous_state
    if previous_state is not None:
        nest.assert_same_structure(previous_state, zero_state)
        previous_state_flat = tf.contrib.framework.nest.flatten(previous_state)
        zero_state_flat = tf.contrib.framework.nest.flatten(zero_state)
        num = len(tf.contrib.framework.nest.flatten(previous_state.cell_state))
        initial_state_flat = previous_state_flat[:num] + zero_state_flat[num:]
        initial_state = tf.contrib.framework.nest.pack_sequence_as(zero_state, initial_state_flat)
    else:
        initial_state = zero_state

    if master_state is not None:
        if sub_bridge is None:
            raise ValueError("A sub_bridge must be configured when passing encoder state")
        else:
            initial_state, master_context_vector = sub_bridge(
                encoder_state=master_state,
                decoder_zero_state=initial_state,
                sub_attention_over_encoder=False)
    else:
        master_context_vector = None

    return initial_state, master_context_vector

def sub_dynamic_decode_and_search(
        decoder,
        sub_emb_gate=None,
        sub_bridge=None,
        beam_width=5,
        master_input=None,
        master_state=None,
        previous_state=None,
        previous_inputs=None,
        impute_finished=False,
        maximum_iterations=None,
        parallel_iterations=32,
        swap_memory=False,
        scope=None,
        dynamic=False,
        force_non_rep=True,
        shifted=None,
        master_attention_at_input=False,
        htm1_at_emb_gate=False):
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

  tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] sub_bridge = {}".format(sub_bridge))
  tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE master_input = {}".format(master_input))
  tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE master_state = {}".format(master_state))
  tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE previous_state = {}".format(previous_state))
  tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] previous_inputs = {}".format(previous_inputs))

  if not isinstance(decoder, tf.contrib.seq2seq.Decoder):
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
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] shifted = {}".format(shifted))

      initial_finished, initial_inputs, initial_state = decoder.initialize()
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] initial_finished = {}".format(initial_finished))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE initial_state = {}".format(initial_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE initial_inputs = {}".format(initial_inputs))

      # ---------------------------------------- preparing for sub decoding ---------------------------------------- #
      batch_size = tf.shape(initial_inputs)[0]
      beam_width = initial_inputs.get_shape().as_list()[-2]

      # previous_state is set to None for master_time = 0
      previous_state = previous_state if isinstance(previous_state, tf.contrib.seq2seq.BeamSearchDecoderState) else None
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER previous_state = {}".format(previous_state))

      initial_state, master_context_vector = _init_sub_state(initial_state,
                                                             sub_bridge,
                                                             previous_state=previous_state,
                                                             master_state=master_state)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER initial_state = {}".format(initial_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE master_context_vector = {}".format(master_context_vector))

      if shifted is not None and previous_inputs is not None:
          initial_inputs = previous_inputs
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER initial_inputs = {}".format(initial_inputs))

      if master_input is not None:
          depth = master_input.get_shape().as_list()[-1]  # get static instead of dynamic shape
          master_input = tf.contrib.seq2seq.tile_batch(master_input, multiplier=beam_width)
          master_input = tf.reshape(master_input, [batch_size, beam_width, depth])
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER master_input = {}".format(master_input))

      if master_context_vector is not None:
          depth = master_context_vector.get_shape().as_list()[-1]  # get static instead of dynamic shape
          master_context_vector = tf.contrib.seq2seq.tile_batch(master_context_vector, multiplier=beam_width)
          master_context_vector = tf.reshape(master_context_vector, [batch_size, beam_width, depth])
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER master_context_vector = {}".format(master_context_vector))

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER master_context_vector = {}".format(master_context_vector))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] maximum_iterations = {}".format(maximum_iterations))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] is_xla = {}".format(is_xla))
      # ---------------------------------------------------- done ---------------------------------------------------- #

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] decoder.output_size = {}".format(decoder.output_size))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] decoder.batch_size = {}".format(decoder.batch_size))
      zero_outputs = _create_zero_outputs(decoder.output_size,
                                          decoder.output_dtype,
                                          decoder.batch_size)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] zero_outputs = {}".format(zero_outputs))

      if is_xla and maximum_iterations is None:
          raise ValueError("maximum_iterations is required for XLA compilation.")
      if maximum_iterations is not None:
          initial_finished = math_ops.logical_or(initial_finished, 0 >= maximum_iterations)

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] initial_finished = {}".format(initial_finished))

      initial_sequence_lengths = array_ops.zeros_like(initial_finished, dtype=dtypes.int32)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] initial_sequence_lengths = {}".format(initial_sequence_lengths))

      initial_time = constant_op.constant(0, dtype=dtypes.int32)

      def _shape(batch_size, from_shape):
          if not isinstance(from_shape, tensor_shape.TensorShape) or from_shape.ndims == 0:
              return tensor_shape.TensorShape(None)
          else:
              batch_size = tensor_util.constant_value(ops.convert_to_tensor(batch_size, name="batch_size"))
              return tensor_shape.TensorShape([batch_size]).concatenate(from_shape)

      dynamic_size = maximum_iterations is None or not is_xla
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] dynamic_size = {}".format(dynamic_size))

      def _create_ta(s, d):
          return tensor_array_ops.TensorArray(
              dtype=d,
              size=0 if dynamic_size else maximum_iterations,
              dynamic_size=dynamic_size,
              element_shape=_shape(decoder.batch_size, s),
              clear_after_read=False)

      initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size, decoder.output_dtype)
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] initial_outputs_ta = {}".format(initial_outputs_ta))

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
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] inputs = {}".format(inputs))
          rnn_inputs = inputs

          if master_input is not None:
              tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] master_input = {}".format(master_input))
              tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] sub_emb_gate = {}".format(sub_emb_gate))
              tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] htm1_at_emb_gate = {}".format(htm1_at_emb_gate))
              tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] callable(sub_emb_gate) = {}".format(callable(sub_emb_gate)))
              master_emb_weight = sub_emb_gate(inputs, state) if htm1_at_emb_gate else sub_emb_gate(inputs)
              tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] master_emb_weight = {}".format(master_emb_weight))
              master_input_weighted = tf.multiply(master_input, master_emb_weight)
              tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] master_input_weighted = {}".format(master_input_weighted))
              rnn_inputs = tf.concat([rnn_inputs, master_input], -1)
          if master_attention_at_input:
              rnn_inputs = tf.concat([rnn_inputs, master_context_vector], -1)
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] rnn_inputs = {}".format(rnn_inputs))

          (next_outputs, decoder_state, next_inputs, decoder_finished) = decoder.step(time, rnn_inputs, state)
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] next_outputs = {}".format(next_outputs))
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] next_inputs = {}".format(next_inputs))
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] decoder_finished = {}".format(decoder_finished))

          if decoder.tracks_own_finished:
              next_finished = decoder_finished
          else:
              next_finished = math_ops.logical_or(decoder_finished, finished)
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] next_finished = {}".format(next_finished))

          next_sequence_lengths = array_ops.where(
              math_ops.logical_not(finished),
              array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
              sequence_lengths)
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] next_sequence_lengths = {}".format(next_sequence_lengths))
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] state = {}".format(state))
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] decoder_state = {}".format(decoder_state))

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
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] emit = {}".format(emit))

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
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] next_state = {}".format(next_state))

          outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out), outputs_ta, emit)

          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] next_finished = {}".format(next_finished))

          if shifted is not None:
              inputs_shape = tf.shape(inputs)
              final_inputs_shape = tf.shape(final_inputs)
              final_inputs = array_ops.where(tf.reshape(next_finished, [-1]),
                                             tf.reshape(inputs, [-1, inputs_shape[-1]]),
                                             tf.reshape(final_inputs, [-1, final_inputs_shape[-1]]))
              final_inputs = tf.reshape(final_inputs, final_inputs_shape)
          tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] final_inputs = {}".format(final_inputs))

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

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] final_inputs = {}".format(final_inputs))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] final_outputs_ta = {}".format(final_outputs_ta))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE final_state = {}".format(final_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] final_sequence_lengths = {}".format(final_sequence_lengths))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] zero_outputs = {}".format(zero_outputs))

      final_outputs = tf.cond(
          math_ops.reduce_all(initial_finished),
          true_fn=lambda: nest.map_structure(lambda t: tf.expand_dims(t, axis=0), zero_outputs),
          false_fn=lambda: nest.map_structure(lambda ta: ta.stack(), final_outputs_ta))

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] final_state = {}".format(final_state))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] final_sequence_lengths = {}".format(final_sequence_lengths))
      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] BEFORE final_outputs = {}".format(final_outputs))

      try:
          final_outputs, final_state = decoder.finalize(final_outputs, final_state, final_sequence_lengths)
      except NotImplementedError:
          pass

      tf.logging.info(" >> [hierarchical_dynamic_beam_search_decode.py sub_dynamic_decode_and_search] AFTER final_outputs = {}".format(final_outputs))

  return final_outputs, final_state, final_sequence_lengths, final_time, final_inputs