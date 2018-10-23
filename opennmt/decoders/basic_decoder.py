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
"""A class of Decoders that may sample to generate the next input.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections,sys
import tensorflow as tf
import numpy as np

from opennmt.decoders import helper as helper_py
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest

from opennmt.decoders.tf_contrib_seq2seq_decoder import TfContribSeq2seqDecoder

__all__ = [
    "BasicDecoderOutput",
    "BasicDecoder",
    "BasicSubDecoder",
]


class BasicDecoderOutput(
    collections.namedtuple("BasicDecoderOutput", ("rnn_output", "sample_id"))):
  pass

class BasicDecoder(TfContribSeq2seqDecoder):
  """Basic sampling decoder."""

  def __init__(self, cell, helper, initial_state, output_layer=None):
    """Initialize BasicDecoder.

    Args:
      cell: An `RNNCell` instance.
      helper: A `Helper` instance.
      initial_state: A (possibly nested tuple of...) tensors and TensorArrays.
        The initial state of the RNNCell.
      output_layer: (Optional) An instance of `tf.layers.Layer`, i.e.,
        `tf.layers.Dense`. Optional layer to apply to the RNN output prior
        to storing the result or sampling.

    Raises:
      TypeError: if `cell`, `helper` or `output_layer` have an incorrect type.
    """
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    if not isinstance(helper, helper_py.Helper):
      raise TypeError("helper must be a Helper, received: %s" % type(helper))
    if (output_layer is not None
        and not isinstance(output_layer, layers_base.Layer)):
      raise TypeError(
          "output_layer must be a Layer, received: %s" % type(output_layer))
    self._cell = cell
    self._helper = helper
    self._initial_state = initial_state
    self._output_layer = output_layer

  @property
  def batch_size(self):
    return self._helper.batch_size

  def _rnn_output_size(self):
    size = self._cell.output_size
    if self._output_layer is None:
      return size
    else:
      # To use layer's compute_output_shape, we need to convert the
      # RNNCell's output_size entries into shapes with an unknown
      # batch size.  We then pass this through the layer's
      # compute_output_shape and read off all but the first (batch)
      # dimensions to get the output size of the rnn with the layer
      # applied to the top.
      output_shape_with_unknown_batch = nest.map_structure(
          lambda s: tensor_shape.TensorShape([None]).concatenate(s),
          size)
      layer_output_shape = self._output_layer.compute_output_shape(
          output_shape_with_unknown_batch)
      return nest.map_structure(lambda s: s[1:], layer_output_shape)

  @property
  def output_size(self):
    # Return the cell output and the id
    return BasicDecoderOutput(
        rnn_output=self._rnn_output_size(),
        sample_id=self._helper.sample_ids_shape)

  @property
  def output_dtype(self):
    # Assume the dtype of the cell is the output_size structure
    # containing the input_state's first component's dtype.
    # Return that structure and the sample_ids_dtype from the helper.
    # [c, h, attention, time, alignments, attention_state]
    dtype = nest.flatten(self._initial_state)[0].dtype
    return BasicDecoderOutput(
        nest.map_structure(lambda _: dtype, self._rnn_output_size()),
        self._helper.sample_ids_dtype)

  def initialize(self, name=None):
    """Initialize the decoder.

    Args:
      name: Name scope for any created operations.

    Returns:
      `(finished, first_inputs, initial_state)`.
    """
    tf.logging.info(" >> [basic_decoder.py BasicDecoder] initialize() : return self._helper.initialize() + (self._initial_state,)")
    return self._helper.initialize() + (self._initial_state,)

  def step(self, time, inputs, state, name=None):
    """Perform a decoding step.

    Args:
      time: scalar `int32` tensor.
      inputs: A (structure of) input tensors.
      state: A (structure of) state tensors and TensorArrays.
      name: Name scope for any created operations.

    Returns:
      `(outputs, next_state, next_inputs, finished)`.
    """
    tf.logging.info(" >> [basic_decoder.py BasicDecoder step] inputs = {}".format(inputs))
    tf.logging.info(">> [basic_decoder.py BasicDecoder step]\n state = {}".format(state))
    with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      tf.logging.info(" >> [basic_decoder.py BasicDecoder step] cell_outputs = {}".format(cell_outputs))
      tf.logging.info(" >> [basic_decoder.py BasicDecoder step] cell_state = {}".format(cell_state))
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs)
      sample_ids = self._helper.sample(
          time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
          time=time,
          outputs=cell_outputs,
          state=cell_state,
          sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return outputs, next_state, next_inputs, finished

  def finalize(self, outputs, final_state, sequence_lengths, sequence_mask_sub):
    raise NotImplementedError

class BasicSubDecoder(BasicDecoder):

    def __init__(self,
                 cell,
                 helper,
                 initial_state,
                 bridge=None,
                 output_layer=None,
                 emb_size=None,
                 sub_attention_over_encoder=False,
                 master_attention_at_output=False,
                 disable_unk=True):
      super(BasicSubDecoder, self).__init__(cell, helper, initial_state, output_layer)
      self._initial_zero_state = initial_state
      self._bridge = bridge
      self._sub_attention_over_encoder = sub_attention_over_encoder
      self._master_attention_at_output = master_attention_at_output
      self._disable_unk = disable_unk
      tf.logging.info(" >> [basic_decoder.py BasicSubDecoder __init__] self._master_attention_at_output = {}".format(self._master_attention_at_output))
      tf.logging.info(" >> [basic_decoder.py BasicSubDecoder __init__] self._output_layer = {}".format(self._output_layer))
      self._master_context_vector = None

      zero_state_flat = tf.contrib.framework.nest.flatten(self._initial_zero_state)
      depth = zero_state_flat[1].get_shape().as_list()[-1]
      tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] depth = {}".format(depth))

      self._emb_gate_layer = tf.layers.Dense(emb_size, activation=tf.sigmoid, use_bias=False, name='embedding_gate')
      # self._emb_gate_layer.build([None, emb_size])
      self._emb_gate_layer.build([None, emb_size+depth])
      tf.logging.info(" >> [basic_decoder.py BasicSubDecoder __init__] self._emb_gate_layer = {}".format(self._emb_gate_layer))
      tf.logging.info(" >> [basic_decoder.py BasicSubDecoder __init__] self._bridge = {}".format(self._bridge))

      if self._master_attention_at_output:
        if not isinstance(self._initial_zero_state, tf.nn.rnn_cell.LSTMStateTuple) or self._sub_attention_over_encoder:
          raise ValueError("master_attention_at_output is only applicable when sub_attention_over_encoder is False")

        # TODO: may need to use activation=tf.tanh
        self._sub_attention_layer = tf.layers.Dense(depth, use_bias=False, name="sub_decoder_attention_layer_dense")
        self._sub_attention_layer.build([None, depth * 2])
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] master_context_vector = {}".format(self._master_context_vector))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] _sub_attention_layer = {}".format(self._sub_attention_layer))

    def _init_sub_state(self, previous_state, master_state=None):
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder _init_sub_state] _sub_attention_over_encoder = {}".format(self._sub_attention_over_encoder))

        if previous_state is None or master_state is None:
            return self._initial_zero_state
        elif master_state is None:
            return previous_state or self._initial_zero_state
        elif self._bridge is None:
            raise ValueError("A sub_bridge must be configured when passing encoder state")

        return self._bridge(encoder_state=master_state,
                            decoder_zero_state=previous_state,
                            sub_attention_over_encoder=self._sub_attention_over_encoder)

    def initialize(self, master_state=None, previous_state=None, master_time=None, name=None):
        """
            initial_state=next_state (master)
            previous_state=sub_state (sub)
        """
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] master_time = {}".format(master_time))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] self._helper = {}".format(self._helper))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] previous_state = {}".format(previous_state))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] BEFORE _initial_state = {}".format(self._initial_state))

        # previous_state is passed as the zero_state for initializing the initial_state for current master_time
        self._initial_state, self._master_context_vector = self._init_sub_state(previous_state=previous_state,
                                                                                master_state=master_state)
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder initialize] AFTER self._initial_state = {}".format(self._initial_state))

        return self._helper.initialize(master_time) + (self._initial_state,) + (self._master_context_vector,)

    def step(self, time, inputs, state, name=None, previous_ids=None, zero_ids=None):
        """Perform a decoding step.

        Args:
          time: scalar `int32` tensor.
          inputs: A (structure of) input tensors.
          state: A (structure of) state tensors and TensorArrays.
          name: Name scope for any created operations.

        Returns:
          `(outputs, next_state, next_inputs, finished)`.
        """
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] inputs = {}".format(inputs))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step]\n state = {}".format(state))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] previous_ids = {}".format(previous_ids))
        tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] zero_ids = {}".format(zero_ids))

        if previous_ids is not None:
            previous_ids = tf.cond(
                tf.equal(time, ops.convert_to_tensor(0, dtype=tf.int32)),
                true_fn=lambda: tf.expand_dims(zero_ids, axis=0),
                false_fn=lambda: previous_ids.stack())
            tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] previous_ids = {}".format(previous_ids))
            previous_ids = tf.transpose(previous_ids, perm=[1, 0])

        with ops.name_scope(name, "BasicDecoderStep", (time, inputs, state)):
            cell_outputs, cell_state = self._cell(inputs, state)
            tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] cell_outputs = {}".format(cell_outputs))
            tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] cell_state = {}".format(cell_state))

            if self._master_attention_at_output:
                if self._master_context_vector is None or self._sub_attention_layer is None:
                    raise ValueError("master_context_vector ({}) or sub_attention_layer ({}) must be available !!!"
                                     .format(self._master_context_vector, self._sub_attention_layer))
                tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] _master_context_vector = {}".format(self._master_context_vector))
                cell_outputs_extended = tf.concat([cell_outputs, self._master_context_vector], axis=-1)
                cell_outputs = self._sub_attention_layer(cell_outputs_extended)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)
            tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] after output_layer cell_outputs = {}".format(cell_outputs))

            if previous_ids is not None:
                previous_ids_one_hot = tf.one_hot(previous_ids,
                                                  depth=cell_outputs.get_shape().as_list()[-1],
                                                  on_value=ops.convert_to_tensor(-np.Inf, dtype=tf.float32),
                                                  off_value=ops.convert_to_tensor(0.0, dtype=tf.float32),
                                                  axis=-1)
                previous_ids_mask = tf.reduce_sum(previous_ids_one_hot, axis=1)
            sample_ids = self._helper.sample(time=time,
                                             outputs=cell_outputs,
                                             state=cell_state,
                                             previous_ids_mask=previous_ids_mask if previous_ids is not None else None,
                                             disable_unk=self._disable_unk)
            tf.logging.info(" >> [basic_decoder.py BasicSubDecoder step] sample_ids = {}".format(sample_ids))

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time,
                outputs=cell_outputs,
                state=cell_state,
                sample_ids=sample_ids)
        outputs = BasicDecoderOutput(cell_outputs, sample_ids)
        return outputs, next_state, next_inputs, finished

    def emb_gate_layer(self, inputs, state):
        state_h = tf.contrib.framework.nest.flatten(state)[1]
        return self._emb_gate_layer(tf.concat([inputs, state_h], axis=-1))
        # return self._emb_gate_layer(inputs)

    @property
    def sub_time(self):
        return self._helper.sub_time if hasattr(self._helper, 'sub_time') else None

    def finalize(self, outputs, final_state, sequence_lengths):
        raise NotImplementedError