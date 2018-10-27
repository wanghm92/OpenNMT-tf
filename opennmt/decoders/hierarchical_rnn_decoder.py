"""Define RNN-based hierarchical decoder."""

import tensorflow as tf
from tensorflow.python.estimator.util import fn_args
import sys

from opennmt.decoders.decoder import build_output_layer, logits_to_cum_log_probs
from opennmt.layers.reducer import align_in_time, align_in_master_time, align_in_master_time_nodepth
from opennmt.decoders.rnn_decoder import RNNDecoder, AttentionalRNNDecoder, _get_alignment_history
from opennmt.decoders.basic_decoder import BasicDecoder, BasicSubDecoder
from opennmt.decoders.tf_contrib_seq2seq_decoder import hierarchical_dynamic_decode
from opennmt.decoders.hierarchical_dynamic_beam_search_decode import hierarchical_dynamic_decode_and_search
from opennmt.decoders.helper import *
from opennmt.utils.misc import add_dict_to_collection

class HierarchicalAttentionalRNNDecoder(AttentionalRNNDecoder):
  """A RNN decoder with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               sub_bridge=None,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               output_is_attention=True,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False,
               pass_master_state=False,
               pass_master_input=False,
               sub_attention_over_encoder=False,
               master_attention_at_input=False,
               master_attention_at_output=False,
               force_non_rep=True,
               disable_unk=True,
               htm1_at_emb_gate=False,
               ):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
      attention_mechanism_class: A class inheriting from
        ``tf.contrib.seq2seq.AttentionMechanism`` or a callable that takes
        ``(num_units, memory, memory_sequence_length)`` as arguments and returns
        a ``tf.contrib.seq2seq.AttentionMechanism``.
      output_is_attention: If ``True``, the final decoder output (before logits)
        is the output of the attention layer. In all cases, the output of the
        attention layer is passed to the next step.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
    super(HierarchicalAttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge=bridge,
        attention_mechanism_class=attention_mechanism_class,
        output_is_attention=output_is_attention,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)
    self._sub_bridge = sub_bridge
    self._pass_master_state = pass_master_state
    self._pass_master_input = pass_master_input
    self._sub_attention_over_encoder = sub_attention_over_encoder
    self._master_attention_at_input = master_attention_at_input
    self._master_attention_at_output = master_attention_at_output
    self._force_non_rep = force_non_rep
    self._disable_unk = disable_unk
    self._htm1_at_emb_gate = htm1_at_emb_gate
    self._sub_emb_gate = None

  def _wrapped_build_cell(self,
                          mode,
                          batch_size,
                          initial_state=None,
                          memory=None,
                          memory_sequence_length=None,
                          dtype=None,
                          alignment_history=False):

    tf.logging.info(
        " >> [rnn_decoder.py class HierarchicalAttentionalRNNDecoder _sub_attention_over_encoder] "
        "_sub_attention_over_encoder = {}".format(self._sub_attention_over_encoder))
    if not self._sub_attention_over_encoder:
        cell, initial_cell_state = RNNDecoder._build_cell(self,
                                                          mode=mode,
                                                          batch_size=batch_size,
                                                          initial_state=initial_state,
                                                          dtype=memory.dtype)
    else:
        cell, initial_cell_state = self._build_cell(mode=mode,
                                                    batch_size=batch_size,
                                                    initial_state=initial_state,
                                                    memory=memory,
                                                    memory_sequence_length=memory_sequence_length,
                                                    dtype=dtype,
                                                    alignment_history=alignment_history)

    tf.logging.info(" >> [rnn_decoder.py class HierarchicalAttentionalRNNDecoder _wrapped_build_cell] initial_cell_state (bridged) = {}"
                    .format(initial_cell_state))
    tf.logging.info(" >> [rnn_decoder.py class HierarchicalAttentionalRNNDecoder _wrapped_build_cell] self.num_units = {}"
                    .format(self.num_units))
    tf.logging.info(" >> [rnn_decoder.py class HierarchicalAttentionalRNNDecoder _wrapped_build_cell] cell = {}"
                    .format(cell))
    return cell, initial_cell_state

  def decode(self,
             inputs,
             sequence_length,
             vocab_size=None,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             output_layer=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None,
             shifted=None):
    """
    Decodes a full input sequence.
    Usually used for training and evaluation where target sequences are known.
    Args:
      inputs = (target_inputs, sub_target_inputs) (embedding lookup)
      sequence_length = self._get_labels_length(labels),
      vocab_size=master_target_vocab_size,
      initial_state=encoder_state,
      sampling_probability=sampling_probability,
      embedding=target_embedding_fn,
      mode=mode,
      memory=encoder_outputs,
      memory_sequence_length=encoder_sequence_length)
    Returns:
      A tuple ``(outputs, state, sequence_length)``.
    """
    _ = memory
    _ = memory_sequence_length

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] memory = {}".format(memory))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] memory_sequence_length = {}".format(memory_sequence_length))

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode]")

    if isinstance(vocab_size, tuple):
        master_vocab_size, sub_vocab_size = vocab_size
    else:
        master_vocab_size = vocab_size
        sub_vocab_size = vocab_size
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] embedding = {}".format(embedding))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] vocab_size = {}".format(vocab_size))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_vocab_size = {}".format(master_vocab_size))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_vocab_size = {}".format(sub_vocab_size))

    master_inputs, sub_inputs = inputs
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_inputs = {}".format(master_inputs))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_inputs = {}".format(sub_inputs))

    master_sequence_length, sub_sequence_length = sequence_length
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_sequence_length = {}".format(master_sequence_length))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_sequence_length = {}".format(sub_sequence_length))

    batch_size = tf.shape(master_inputs)[0]
    master_time = tf.shape(master_inputs)[1]  # [batch, mt, dim]

    sub_inputs = align_in_master_time(sub_inputs, master_time)
    sub_sequence_length = align_in_master_time_nodepth(sub_sequence_length, master_time)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] AFTER sub_inputs = {}".format(sub_inputs))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] AFTER sub_sequence_length = {}".format(sub_sequence_length))

    if (sampling_probability is not None
        and (tf.contrib.framework.is_tensor(sampling_probability)
             or sampling_probability > 0.0)):
        if embedding is None:
            raise ValueError("embedding argument must be set when using scheduled sampling")
        elif isinstance(embedding, tuple):
            master_embedding, sub_embedding = embedding
        else:
            master_embedding = embedding
            sub_embedding = embedding

        tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_embedding = {}".format(master_embedding))
        tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_embedding = {}".format(sub_embedding))

        tf.summary.scalar("sampling_probability", sampling_probability)
        '''
          A training helper that adds scheduled sampling.
          Returns -1s for sample_ids where no sampling took place; valid sample id values elsewhere.
        '''
        master_helper = ScheduledEmbeddingTrainingHelper(
            master_inputs,
            master_sequence_length,
            master_embedding,
            sampling_probability)

        # sub_helper = ScheduledEmbeddingTrainingHelper(
        #     sub_inputs,
        #     sub_sequence_length,
        #     sub_embedding,
        #     sampling_probability)
        fused_projection = False
    else:
        '''
          A helper for use during training. Only reads inputs.
          Returned sample_ids are the argmax of the RNN output logits.
        '''
        master_helper = TrainingHelper(master_inputs, master_sequence_length)
        fused_projection = True

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] fused_projection = {}".format(fused_projection))

    sub_helper = HierarchicalTrainingHelper(sub_inputs, sub_sequence_length)
    fused_projection_sub = True

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_helper = {}".format(master_helper))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_helper = {}".format(sub_helper))


    '''
    Pass memory and initial_state to _build_cell() when building AttentionalRNNDecoder
    memory = encoder_outputs
    initial_state = encoder_state 
        LSTMStateTuple(c=<tf.Tensor 'seq2seq/parallel_0/seq2seq/encoder/concat_1:0' shape=(?, 128) dtype=float32>, 
                       h=<tf.Tensor 'seq2seq/parallel_0/seq2seq/encoder/concat_2:0' shape=(?, 128) dtype=float32>)
    '''

    master_cell, initial_state_master = self._build_cell(
        mode=mode,
        batch_size=batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=master_inputs.dtype)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_cell = {}".format(master_cell))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] initial_state_master = {}".format(initial_state_master))

    sub_cell, initial_state_sub = self._wrapped_build_cell(
        mode=mode,
        batch_size=batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=sub_inputs.dtype)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_cell = {}".format(sub_cell))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] initial_state_sub = {}".format(initial_state_sub))

    if output_layer is None:
        master_output_layer = build_output_layer(self.num_units, master_vocab_size, dtype=master_inputs.dtype, name="master_output_layer")
        sub_output_layer = build_output_layer(self.num_units, sub_vocab_size, dtype=master_inputs.dtype, name="sub_output_layer")
    else:
        if not isinstance(output_layer, tuple):
            raise ValueError("Two separate output_layer needed if not None")
        master_output_layer, sub_output_layer = output_layer

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] master_output_layer = {}".format(master_output_layer))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] sub_output_layer = {}".format(sub_output_layer))

    '''
    master and sub-sequence sampling decoder.
    '''
    master_decoder = BasicDecoder(
        cell=master_cell,
        helper=master_helper,
        initial_state=initial_state_master,
        output_layer=master_output_layer if not fused_projection else None)

    emb_size = sub_inputs.get_shape().as_list()[-1]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] emb_size = {}".format(emb_size))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] _master_attention_at_output = {}".format(self._master_attention_at_output))
    sub_decoder = BasicSubDecoder(
        cell=sub_cell,
        helper=sub_helper,
        initial_state=initial_state_sub,
        bridge=self._sub_bridge,
        output_layer=sub_output_layer if not fused_projection_sub else None,
        emb_size=emb_size,
        sub_attention_over_encoder=self._sub_attention_over_encoder,
        master_attention_at_output=self._master_attention_at_output,
        disable_unk=self._disable_unk,
        htm1_at_emb_gate=self._htm1_at_emb_gate)

    self._sub_emb_gate = sub_decoder.emb_gate_layer
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] self._sub_emb_gate = {}".format(self._sub_emb_gate))

    '''
    Perform dynamic decoding with decoder.
    Calls initialize() once and step() repeatedly on the Decoder object.
    Returns:
    (final_outputs, final_state, final_sequence_lengths).
    state is final RNN state (last time step)
    outputs is all RNN outputs (all time steps)
        outputs, state, length = tf.contrib.seq2seq.dynamic_decode(decoder)
    '''
    outputs, outputs_sub, state, state_sub, length, sequence_mask_sub, final_time = hierarchical_dynamic_decode(
                                                                                        master_decoder,
                                                                                        sub_decoder,
                                                                                        shifted=shifted,
                                                                                        pass_master_state=self._pass_master_state,
                                                                                        pass_master_input=self._pass_master_input,
                                                                                        master_attention_at_input=self._master_attention_at_input,
                                                                                        htm1_at_emb_gate=self._htm1_at_emb_gate)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] outputs = {}".format(outputs))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] outputs.rnn_output = {}".format(outputs))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] outputs_sub = {}".format(outputs_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] outputs_sub.rnn_output = {}".format(outputs_sub.rnn_output))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] state = {}".format(state))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] length = {}".format(length))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] final_time = {}".format(final_time))

    if fused_projection and master_output_layer is not None:
        logits = master_output_layer(outputs.rnn_output)
    else:
        logits = outputs.rnn_output
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] BEFORE logits = {}".format(logits))
    # Make sure outputs have the same time_dim as inputs
    logits = align_in_time(logits, master_time)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] AFTER logits = {}".format(logits))

    if fused_projection_sub and sub_output_layer is not None:
        logits_sub = sub_output_layer(outputs_sub.rnn_output)
    else:
        logits_sub = outputs_sub.rnn_output
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] BEFORE logits_sub = {}".format(logits_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] BEFORE sequence_mask_sub = {}".format(sequence_mask_sub))

    # [batch, mt, st, depth], [batch, mt, st]
    # logits_sub = align_in_time_2d(logits_sub, master_time, tf.shape(sub_inputs)[2])
    logits_sub = align_in_master_time(logits_sub, master_time)
    sequence_mask_sub = align_in_time(sequence_mask_sub, master_time)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] AFTER logits_sub = {}".format(logits_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] AFTER sequence_mask_sub = {}".format(sequence_mask_sub))

    add_dict_to_collection("debug", {"decode_final_time": final_time,
                                     "decode_sequence_mask_sub_shape": tf.shape(sequence_mask_sub),
                                     "decode_outputs.rnn_output shape": tf.shape(outputs.rnn_output),
                                     "decode_outputs_sub.rnn_output shape": tf.shape(outputs_sub.rnn_output),
                                     # "decode_cell_state_c": tf.shape(state.cell_state.c),
                                     # "decode_cell_state_h": tf.shape(state.cell_state.h),
                                     "decode_attention": tf.shape(state.attention),
                                     "decode_time": state.time,
                                     "decode_alignments": tf.shape(state.alignments),
                                     "decode_inputs_len": master_time,
                                     "decode_logits": tf.shape(logits),
                                     "decode_logits_sub": tf.shape(logits_sub),
                                     "decode_sequence_mask_sub": tf.shape(sequence_mask_sub),
                                     "decode_master_sequence_length": master_sequence_length,
                                     "decode_sub_sequence_length": sub_sequence_length,
                                     })

    return (logits, logits_sub, state, length, sequence_mask_sub)

  def dynamic_decode(self,
                     embedding,
                     start_tokens,
                     end_token,
                     vocab_size=None,
                     initial_state=None,
                     output_layer=None,
                     maximum_iterations=None,
                     sub_maximum_iterations=None,
                     mode=tf.estimator.ModeKeys.PREDICT,
                     memory=None,
                     memory_sequence_length=None,
                     dtype=None,
                     return_alignment_history=False,
                     shifted=None):
    """Decodes dynamically from :obj:`start_tokens` with greedy search.

    Usually used for inference. (decode() uses TrainingHelpers)
    Args:
      embedding=target_embedding_fn,
      start_tokens=start_tokens,
      end_token=end_token,
      vocab_size=target_vocab_size,
      initial_state=encoder_state,
      maximum_iterations=maximum_iterations,
      mode=mode,
      memory=encoder_outputs,
      memory_sequence_length=encoder_sequence_length,
      dtype=target_dtype,
      return_alignment_history=True)
    Returns:
      A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
      ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
      if :obj:`return_alignment_history` is ``True``.
    """

    if isinstance(embedding, tuple):
      master_embedding, sub_embedding = embedding
    else:
      master_embedding = embedding
      sub_embedding = embedding


    if isinstance(vocab_size, tuple):
      master_vocab_size, sub_vocab_size = vocab_size
    else:
      master_vocab_size = vocab_size
      sub_vocab_size = vocab_size

    batch_size = tf.shape(start_tokens)[0]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] batch_size = {}".format(batch_size))

    master_helper = GreedyEmbeddingHelper(master_embedding, start_tokens, end_token)
    sub_helper = HierarchicalGreedyEmbeddingHelper(sub_embedding, start_tokens, end_token)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] master_helper = {}".format(master_helper))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] sub_helper = {}".format(sub_helper))

    master_cell, initial_state_master = self._build_cell(
        mode=mode,
        batch_size=batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype,
        alignment_history=return_alignment_history)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] master_cell = {}".format(master_cell))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] initial_state_master = {}".format(initial_state_master))

    sub_cell, initial_state_sub = self._wrapped_build_cell(
        mode=mode,
        batch_size=batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype,
        alignment_history=return_alignment_history)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] sub_cell = {}".format(sub_cell))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] initial_state_sub = {}".format(initial_state_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] output_layer = {}".format(output_layer))

    if output_layer is None:
        master_output_layer = build_output_layer(self.num_units, master_vocab_size, dtype=dtype or memory.dtype, name="master_output_layer")
        sub_output_layer = build_output_layer(self.num_units, sub_vocab_size, dtype=dtype or memory.dtype, name="sub_output_layer")
    else:
        if not isinstance(output_layer, tuple):
            raise ValueError("Two separate output_layer needed if not None")
        master_output_layer, sub_output_layer = output_layer

    master_decoder = BasicDecoder(
        cell=master_cell,
        helper=master_helper,
        initial_state=initial_state_master,
        output_layer=master_output_layer)

    emb_size = sub_helper._start_inputs.get_shape().as_list()[-1]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] emb_size = {}".format(emb_size))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] _master_attention_at_output = {}".format(self._master_attention_at_output))
    sub_decoder = BasicSubDecoder(
        cell=sub_cell,
        helper=sub_helper,
        initial_state=initial_state_sub,
        bridge=self._sub_bridge,
        output_layer=sub_output_layer,
        emb_size=emb_size,
        sub_attention_over_encoder=self._sub_attention_over_encoder,
        master_attention_at_output=self._master_attention_at_output,
        disable_unk=self._disable_unk,
        htm1_at_emb_gate=self._htm1_at_emb_gate)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py decode] self._sub_emb_gate = {}".format(self._sub_emb_gate))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] master_decoder = {}".format(master_decoder))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] sub_decoder = {}".format(sub_decoder))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] maximum_iterations = {}".format(maximum_iterations))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] sub_maximum_iterations = {}".format(sub_maximum_iterations))

    outputs, outputs_sub, state_master, state_sub, length, sequence_mask_sub, final_time = hierarchical_dynamic_decode(
                                                                                                master_decoder,
                                                                                                sub_decoder,
                                                                                                maximum_iterations=maximum_iterations,
                                                                                                sub_maximum_iterations=sub_maximum_iterations,
                                                                                                dynamic=True,
                                                                                                force_non_rep=self._force_non_rep,
                                                                                                shifted=shifted,
                                                                                                pass_master_state=self._pass_master_state,
                                                                                                pass_master_input=self._pass_master_input,
                                                                                                master_attention_at_input=self._master_attention_at_input,
                                                                                                htm1_at_emb_gate=self._htm1_at_emb_gate)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] outputs = {}".format(outputs))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] outputs.rnn_output = {}".format(outputs.rnn_output))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] outputs_sub = {}".format(outputs_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] outputs_sub.rnn_output = {}".format(outputs_sub.rnn_output)) # [batch, sum_of_st, depth]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] state_master = {}".format(state_master))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] state_sub = {}".format(state_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] length = {}".format(length))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] sequence_mask_sub = {}".format(sequence_mask_sub)) # [batch, sum_of_st]

    master_predicted_ids = outputs.sample_id
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] master_predicted_ids = {}".format(master_predicted_ids))
    sub_predicted_ids = outputs_sub.sample_id
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] BEFORE sub_predicted_ids = {}".format(sub_predicted_ids))

    # mask out unwanted output tokens
    sub_predicted_ids = sub_predicted_ids * tf.cast(sequence_mask_sub, sub_predicted_ids.dtype)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] AFTER sub_predicted_ids = {}".format(sub_predicted_ids)) # [batch, sum_of_st]

    # NOTE: sub_length is not used for final output written to file, due to <blank> tokens in between
    sub_length = tf.reduce_sum(sequence_mask_sub, axis=-1)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] sub_length = {}".format(sub_length))

    log_probs_master = logits_to_cum_log_probs(outputs.rnn_output, length)
    log_probs_sub = logits_to_cum_log_probs(outputs_sub.rnn_output, sequence_mask_sub)
    log_probs = log_probs_master + log_probs_sub

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] log_probs = {}".format(log_probs_master))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode] log_probs_sub = {}".format(log_probs_sub)) # [batch, sum_of_st, depth]
    add_dict_to_collection("debug", {"eval_log_probs_master": log_probs_master,
                                     "eval_log_probs_sub": log_probs_sub,
                                     })

    # Make shape consistent with beam search.
    master_predicted_ids = tf.expand_dims(master_predicted_ids, 1)
    sub_predicted_ids = tf.expand_dims(sub_predicted_ids, 1)
    master_length = tf.expand_dims(length, 1)
    sub_length = tf.cast(tf.expand_dims(sub_length, 1), tf.int64)
    # NOTE: this may not be consistent with beam search
    log_probs = tf.expand_dims(log_probs, 1)

    predicted_ids = (master_predicted_ids, sub_predicted_ids)
    length = (master_length, sub_length)

    state_tuple = (state_master, state_sub)
    if return_alignment_history:
      alignment_history = _get_alignment_history(state_master)
      if alignment_history is not None:
        alignment_history = tf.expand_dims(alignment_history, 1)

      alignment_history_sub = _get_alignment_history(state_sub)
      if alignment_history_sub is not None:
        alignment_history_sub = tf.expand_dims(alignment_history_sub, 1)
      alignment_history = (alignment_history, alignment_history_sub)

      return (predicted_ids, state_tuple, length, log_probs, alignment_history)

    return (predicted_ids, state_tuple, length, log_probs)

  def dynamic_decode_and_search(self,
                                embedding,
                                start_tokens,
                                end_token,
                                vocab_size=None,
                                initial_state=None,
                                output_layer=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=None,
                                sub_maximum_iterations=None,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None,
                                dtype=None,
                                return_alignment_history=False,
                                shifted=None):
    """Decodes dynamically from :obj:`start_tokens` with beam search.

    Usually used for inference.
    Args:
      embedding=target_embedding_fn,
      start_tokens=start_tokens,
      end_token=end_token,
      vocab_size=target_vocab_size,
      initial_state=encoder_state,
      beam_width=beam_width,
      length_penalty=length_penalty,
      maximum_iterations=maximum_iterations,
      mode=mode,
      memory=encoder_outputs,
      memory_sequence_length=encoder_sequence_length,
      dtype=target_dtype,
      return_alignment_history=True))
    Returns:
      A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
      ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
      if :obj:`return_alignment_history` is ``True``.
    """
    # -------------------------------------- alignment history flag and batch_size ----------------------------------- #
    if (return_alignment_history and
        "reorder_tensor_arrays" not in fn_args(tf.contrib.seq2seq.BeamSearchDecoder.__init__)):
      tf.logging.warn("The current version of tf.contrib.seq2seq.BeamSearchDecoder "
                      "does not support returning the alignment history. None will "
                      "be returned instead. Consider upgrading TensorFlow.")
      alignment_history = False
    else:
      alignment_history = return_alignment_history
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] alignment_history = {}".format(alignment_history))

    batch_size = tf.shape(start_tokens)[0]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] batch_size = {}".format(batch_size))

    # -------------------------------------------------- Build cells ------------------------------------------------- #
    master_cell, initial_state_master = self._build_cell(
        mode=mode,
        batch_size=batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype,
        alignment_history=alignment_history)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] master_cell = {}".format(master_cell))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] initial_state_master = {}".format(initial_state_master))

    # ---------------------------------------- Replicate batch `beam_width` times. ----------------------------------- #
    # TODO: only designed to do beam search on sub decoder
    if initial_state is not None:
      initial_state = tf.contrib.seq2seq.tile_batch(
          initial_state, multiplier=beam_width)
    if memory is not None:
      memory = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=beam_width)
    if memory_sequence_length is not None:
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(
          memory_sequence_length, multiplier=beam_width)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] initial_state = {}".format(initial_state))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] memory = {}".format(memory))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] memory_sequence_length = {}".format(memory_sequence_length))

    sub_cell, initial_state_sub = self._wrapped_build_cell(
        mode=mode,
        batch_size=batch_size * beam_width,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype,
        alignment_history=alignment_history)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] sub_cell = {}".format(sub_cell))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] initial_state_sub = {}".format(initial_state_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] output_layer = {}".format(output_layer))

    # -------------------------------------------------- Output layers ----------------------------------------------- #
    if isinstance(embedding, tuple):
      master_embedding, sub_embedding = embedding
    else:
      master_embedding = embedding
      sub_embedding = embedding
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] master_embedding = {}".format(master_embedding))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] sub_embedding = {}".format(sub_embedding))

    if isinstance(vocab_size, tuple):
      master_vocab_size, sub_vocab_size = vocab_size
    else:
      master_vocab_size = vocab_size
      sub_vocab_size = vocab_size
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] master_vocab_size = {}".format(master_vocab_size))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] sub_vocab_size = {}".format(sub_vocab_size))

    if output_layer is None:
        master_output_layer = build_output_layer(self.num_units, master_vocab_size, dtype=dtype or memory.dtype, name="master_output_layer")
        sub_output_layer = build_output_layer(self.num_units, sub_vocab_size, dtype=dtype or memory.dtype, name="sub_output_layer")
    else:
        if not isinstance(output_layer, tuple):
            raise ValueError("Two separate output_layer needed if not None")
        master_output_layer, sub_output_layer = output_layer

    master_helper = GreedyEmbeddingHelper(master_embedding, start_tokens, end_token)
    master_decoder = BasicDecoder(
        cell=master_cell,
        helper=master_helper,
        initial_state=initial_state_master,
        output_layer=master_output_layer)

    sub_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell=sub_cell,
        embedding=sub_embedding,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=initial_state_sub,
        beam_width=beam_width,
        output_layer=sub_output_layer,
        length_penalty_weight=length_penalty)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] self._sub_emb_gate = {}".format(self._sub_emb_gate))

    outputs, outputs_beam_sub, state_master, beam_state_sub, length, sequence_mask_sub, final_time = hierarchical_dynamic_decode_and_search(
                                                                                                        master_decoder,
                                                                                                        sub_decoder,
                                                                                                        sub_emb_gate=self._sub_emb_gate,
                                                                                                        sub_bridge=self._sub_bridge,
                                                                                                        beam_width=beam_width,
                                                                                                        maximum_iterations=maximum_iterations,
                                                                                                        sub_maximum_iterations=sub_maximum_iterations,
                                                                                                        dynamic=True,
                                                                                                        force_non_rep=self._force_non_rep,
                                                                                                        shifted=shifted,
                                                                                                        pass_master_state=self._pass_master_state,
                                                                                                        pass_master_input=self._pass_master_input,
                                                                                                        master_attention_at_input=self._master_attention_at_input,
                                                                                                        htm1_at_emb_gate=self._htm1_at_emb_gate)

    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] state_master = {}".format(state_master))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] outputs = {}".format(outputs))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] outputs_beam_sub = {}".format(outputs_beam_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] beam_state_sub = {}".format(beam_state_sub))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] length = {}".format(length))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] BEFORE sequence_mask_sub = {}".format(sequence_mask_sub))

    state_sub = beam_state_sub.cell_state
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] state_sub = {}".format(state_sub))

    master_predicted_ids = outputs.sample_id
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] master_predicted_ids = {}".format(master_predicted_ids))

    sub_predicted_ids = tf.transpose(outputs_beam_sub.predicted_ids, perm=[0, 2, 1])  # [batch, beam_size, sum_of_st]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] BEFORE sub_predicted_ids = {}".format(sub_predicted_ids))  # [batch, beam_size, sum_of_st]

    # mask out unwanted output tokens
    sequence_mask_sub = tf.transpose(sequence_mask_sub, perm=[0, 2, 1])  # [batch, beam_size, sum_of_st]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] AFTER sequence_mask_sub = {}".format(sequence_mask_sub))
    sub_predicted_ids = sub_predicted_ids * tf.cast(sequence_mask_sub, sub_predicted_ids.dtype)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] AFTER sub_predicted_ids = {}".format(sub_predicted_ids))  # [batch, beam_size, sum_of_st]

    # NOTE: sub_length is not used for final output written to file, due to <blank> tokens in between
    sub_length = tf.reduce_sum(sequence_mask_sub, axis=-1)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] sub_length = {}".format(sub_length))

    log_probs_master = logits_to_cum_log_probs(outputs.rnn_output, length)
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] outputs.rnn_output = {}".format(outputs.rnn_output))
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] log_probs_master = {}".format(log_probs_master))

    log_probs_sub = beam_state_sub.log_probs  # [batch, beam_size]
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] log_probs_sub = {}".format(log_probs_sub))

    # Make shape consistent with beam search.
    master_predicted_ids = tf.expand_dims(master_predicted_ids, 1)
    master_length = tf.expand_dims(length, 1)
    sub_length = tf.cast(tf.expand_dims(sub_length, 1), tf.int64)
    log_probs_master = tf.expand_dims(log_probs_master, 1)
    log_probs = log_probs_master + log_probs_sub
    tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] log_probs = {}".format(log_probs_master))

    predicted_ids = (master_predicted_ids, sub_predicted_ids)
    length = (master_length, sub_length)
    state_tuple = (state_master, state_sub)

    if return_alignment_history:
      alignment_history = _get_alignment_history(state_master)
      if alignment_history is not None:
        alignment_history = tf.expand_dims(alignment_history, 1)

      alignment_history_sub = _get_alignment_history(state_sub)
      if alignment_history_sub is not None:
        alignment_history_sub = tf.reshape(
            alignment_history_sub, [batch_size, beam_width, -1, tf.shape(memory)[1]])
      alignment_history = (alignment_history, alignment_history_sub)

      return (predicted_ids, state_tuple, length, log_probs, alignment_history)

    return (predicted_ids, state_tuple, length, log_probs)



    # -------------------------------------------------- Decoders ---------------------------------------------------- #
    # master_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
    #     master_cell,
    #     master_embedding,
    #     start_tokens,
    #     end_token,
    #     initial_state_master,
    #     beam_width,
    #     output_layer=master_output_layer,
    #     length_penalty_weight=length_penalty)
    #
    # outputs, beam_state, length = tf_contrib_seq2seq_dynamic_decode(
    #     master_decoder, maximum_iterations=maximum_iterations)
    #
    # predicted_ids = tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])
    # log_probs = beam_state.log_probs
    # state = beam_state.cell_state
    #
    # tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] outputs = {}".format(outputs))
    # tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] beam_state = {}".format(beam_state))
    # tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] predicted_ids = {}".format(predicted_ids))
    # tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] log_probs = {}".format(log_probs))
    # tf.logging.info(" >> [hierarchical_rnn_decoder.py dynamic_decode_and_search] state = {}".format(state))