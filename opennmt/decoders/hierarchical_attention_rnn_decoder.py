"""Define RNN-based hierarchical decoder."""

import tensorflow as tf

from opennmt.decoders.decoder import build_output_layer
from opennmt.layers.reducer import align_in_time
from opennmt.decoders.rnn_decoder import RNNDecoder, AttentionalRNNDecoder, _build_attention_mechanism
from opennmt.decoders.basic_decoder import BasicDecoder
from opennmt.decoders.hierarchical_dynamic_decoder import hierarchical_dynamic_decode

class HierAttRNNDecoder(AttentionalRNNDecoder):
  """A RNN decoder with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               output_is_attention=True,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
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
    super(HierAttRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge=bridge,
        attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
        output_is_attention=True,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)

  def decode(self,
             inputs,
             sequence_length,
             vocab_size_master=None,
             vocab_size_sub=None,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             output_layer_master=None,
             output_layer_sub=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None):
    """
    Decodes a full input sequence.
    Usually used for training and evaluation where target sequences are known.
    Args:
      inputs = target_inputs, (embedding lookup)
      sequence_length = self._get_labels_length(labels),
      vocab_size=target_vocab_size,
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

    tf.logging.info(" >> [rnn_decoder.py decode]")

    batch_size = tf.shape(inputs)[0]

    # TODO: 'inputs' should be able to split into 2 levels of information (multiple pieces !!!!!!)
    # TODO: 'sequence_length' should be able to split into 2 levels of information
    # TODO: 'vocab_size' should be 2 vocabs, one for attributes, one for words
    vocab_size_sub = vocab_size_master
    # TODO: 'embedding' should also be 2

    if (sampling_probability is not None
        and (tf.contrib.framework.is_tensor(sampling_probability)
             or sampling_probability > 0.0)):
        if embedding is None:
            raise ValueError("embedding argument must be set when using scheduled sampling")

        tf.summary.scalar("sampling_probability", sampling_probability)
        '''
          A training helper that adds scheduled sampling.
          Returns -1s for sample_ids where no sampling took place; valid sample id values elsewhere.
        '''
        master_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs,
            sequence_length,
            embedding,
            sampling_probability)

        # TODO: specify this
        sub_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs,
            sequence_length,
            embedding,
            sampling_probability)
        fused_projection = False
    else:
        '''
          A helper for use during training. Only reads inputs.
          Returned sample_ids are the argmax of the RNN output logits.
        '''
        master_helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)

        # TODO: specify this
        sub_helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)

        fused_projection = True

    '''
    Pass memory and initial_state to _build_cell() when building AttentionalRNNDecoder
    memory = encoder_outputs
    initial_state = encoder_state 
        LSTMStateTuple(c=<tf.Tensor 'seq2seq/parallel_0/seq2seq/encoder/concat_1:0' shape=(?, 128) dtype=float32>, 
                       h=<tf.Tensor 'seq2seq/parallel_0/seq2seq/encoder/concat_2:0' shape=(?, 128) dtype=float32>)
    '''
    master_cell, initial_state_master = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=inputs.dtype)

    # TODO: specify this
    # TODO: remember to wrap using AttentionWrapper()
    sub_cell, initial_state_sub = RNNDecoder._build_cell(
        self,
        mode,
        batch_size,
        initial_state=initial_state,
        dtype=inputs.dtype)

    print('\n')
    print('*' * 10 + 'inputs' + '*' * 10)
    print(inputs)
    print('*' * 10 + 'initial_state_master' + '*' * 10)
    print(initial_state_master)
    print('*' * 10 + 'master_cell' + '*' * 10)
    print(master_cell)
    print('*' * 10 + 'initial_state_sub' + '*' * 10)
    print(initial_state_sub)
    print('*' * 10 + 'sub_cell' + '*' * 10)
    print(sub_cell)
    print('\n')

    if output_layer_master is None:
        output_layer_master = build_output_layer(self.num_units, vocab_size_master, dtype=inputs.dtype)
    if output_layer_sub is None:
        output_layer_sub = build_output_layer(self.num_units, vocab_size_sub, dtype=inputs.dtype)

    '''
    master and sub-sequence sampling decoder.
    '''
    master_decoder = BasicDecoder(
        master_cell,
        master_helper,
        initial_state_master,
        output_layer=output_layer_master if not fused_projection else None)

    sub_decoder = BasicDecoder(
        sub_cell,
        sub_helper,
        initial_state_sub,
        output_layer=output_layer_sub if not fused_projection else None)

    '''
    Perform dynamic decoding with decoder.
    Calls initialize() once and step() repeatedly on the Decoder object.
    Returns:
    (final_outputs, final_state, final_sequence_lengths).
    state is final RNN state (last time step)
    outputs is all RNN outputs (all time steps)
        outputs, state, length = tf.contrib.seq2seq.dynamic_decode(decoder)
    '''
    outputs, state, length = hierarchical_dynamic_decode(master_decoder, sub_decoder)

    if fused_projection and output_layer_master is not None:
        logits = output_layer_master(outputs.rnn_output)
    else:
        logits = outputs.rnn_output
    # Make sure outputs have the same time_dim as inputs
    inputs_len = tf.shape(inputs)[1]
    logits = align_in_time(logits, inputs_len)

    return (logits, outputs.rnn_output, state, length)

  # def dynamic_decode(self,
  #                embedding,
  #                start_tokens,
  #                end_token,
  #                vocab_size=None,
  #                initial_state=None,
  #                output_layer=None,
  #                maximum_iterations=250,
  #                mode=tf.estimator.ModeKeys.PREDICT,
  #                memory=None,
  #                memory_sequence_length=None,
  #                dtype=None,
  #                return_alignment_history=False):
  #   """Decodes dynamically from :obj:`start_tokens` with greedy search.
  #
  #   Usually used for inference. (decode use TrainingHelpers)
  #   Returns:
  #     A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
  #     ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
  #     if :obj:`return_alignment_history` is ``True``.
  #   """
  #
  #   batch_size = tf.shape(start_tokens)[0]
  #
  #   helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
  #       embedding,
  #       start_tokens,
  #       end_token)
  #
  #   cell, initial_state = self._build_cell(
  #       mode,
  #       batch_size,
  #       initial_state=initial_state,
  #       memory=memory,
  #       memory_sequence_length=memory_sequence_length,
  #       dtype=dtype,
  #       alignment_history=return_alignment_history)
  #
  #   if output_layer is None:
  #     output_layer = build_output_layer(self.num_units, vocab_size, dtype=dtype or memory.dtype)
  #
  #   decoder = tf.contrib.seq2seq.BasicDecoder(
  #       cell,
  #       helper,
  #       initial_state,
  #       output_layer=output_layer)
  #
  #   outputs, state, length = tf.contrib.seq2seq.dynamic_decode(
  #       decoder, maximum_iterations=maximum_iterations)
  #
  #   predicted_ids = outputs.sample_id
  #   log_probs = logits_to_cum_log_probs(outputs.rnn_output, length)
  #
  #   # Make shape consistent with beam search.
  #   predicted_ids = tf.expand_dims(predicted_ids, 1)
  #   length = tf.expand_dims(length, 1)
  #   log_probs = tf.expand_dims(log_probs, 1)
  #
  #   if return_alignment_history:
  #     alignment_history = _get_alignment_history(state)
  #     if alignment_history is not None:
  #       alignment_history = tf.expand_dims(alignment_history, 1)
  #     return (predicted_ids, state, length, log_probs, alignment_history)
  #   return (predicted_ids, state, length, log_probs)
  #
  # def dynamic_decode_and_search(self,
  #                               embedding,
  #                               start_tokens,
  #                               end_token,
  #                               vocab_size=None,
  #                               initial_state=None,
  #                               output_layer=None,
  #                               beam_width=5,
  #                               length_penalty=0.0,
  #                               maximum_iterations=250,
  #                               mode=tf.estimator.ModeKeys.PREDICT,
  #                               memory=None,
  #                               memory_sequence_length=None,
  #                               dtype=None,
  #                               return_alignment_history=False):
  #   """Decodes dynamically from :obj:`start_tokens` with beam search.
  #
  #   Usually used for inference.
  #   Returns:
  #     A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
  #     ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
  #     if :obj:`return_alignment_history` is ``True``.
  #   """
  #
  #   if (return_alignment_history and
  #       "reorder_tensor_arrays" not in fn_args(tf.contrib.seq2seq.BeamSearchDecoder.__init__)):
  #     tf.logging.warn("The current version of tf.contrib.seq2seq.BeamSearchDecoder "
  #                     "does not support returning the alignment history. None will "
  #                     "be returned instead. Consider upgrading TensorFlow.")
  #     alignment_history = False
  #   else:
  #     alignment_history = return_alignment_history
  #
  #   batch_size = tf.shape(start_tokens)[0]
  #
  #   # Replicate batch `beam_width` times.
  #   if initial_state is not None:
  #     initial_state = tf.contrib.seq2seq.tile_batch(
  #         initial_state, multiplier=beam_width)
  #   if memory is not None:
  #     memory = tf.contrib.seq2seq.tile_batch(
  #         memory, multiplier=beam_width)
  #   if memory_sequence_length is not None:
  #     memory_sequence_length = tf.contrib.seq2seq.tile_batch(
  #         memory_sequence_length, multiplier=beam_width)
  #
  #   cell, initial_state = self._build_cell(
  #       mode,
  #       batch_size * beam_width,
  #       initial_state=initial_state,
  #       memory=memory,
  #       memory_sequence_length=memory_sequence_length,
  #       dtype=dtype,
  #       alignment_history=alignment_history)
  #
  #   if output_layer is None:
  #     output_layer = build_output_layer(self.num_units, vocab_size, dtype=dtype or memory.dtype)
  #
  #   tf.logging.info(" >> [sequence_to_sequence.py _build] tf.contrib.seq2seq.BeamSearchDecoder ...")
  #   decoder = tf.contrib.seq2seq.BeamSearchDecoder(
  #       cell,
  #       embedding,
  #       start_tokens,
  #       end_token,
  #       initial_state,
  #       beam_width,
  #       output_layer=output_layer,
  #       length_penalty_weight=length_penalty)
  #
  #   tf.logging.info(" >> [sequence_to_sequence.py _build] tf.contrib.seq2seq.dynamic_decode ...")
  #   outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(
  #       decoder, maximum_iterations=maximum_iterations)
  #
  #   predicted_ids = tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])
  #   log_probs = beam_state.log_probs
  #   state = beam_state.cell_state
  #
  #   if return_alignment_history:
  #     alignment_history = _get_alignment_history(state)
  #     if alignment_history is not None:
  #       alignment_history = tf.reshape(
  #           alignment_history, [batch_size, beam_width, -1, tf.shape(memory)[1]])
  #     return (predicted_ids, state, length, log_probs, alignment_history)
  #   return (predicted_ids, state, length, log_probs)