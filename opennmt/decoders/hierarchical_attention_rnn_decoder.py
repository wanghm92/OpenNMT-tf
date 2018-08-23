"""Define RNN-based hierarchical decoder."""

import tensorflow as tf

from opennmt.decoders.decoder import build_output_layer
from opennmt.layers.reducer import align_in_time
from opennmt.decoders.rnn_decoder import RNNDecoder, AttentionalRNNDecoder, _build_attention_mechanism
from opennmt.decoders.basic_decoder import BasicDecoder, BasicSubDecoder
from opennmt.decoders.hierarchical_dynamic_decoder import hierarchical_dynamic_decode
from opennmt.decoders.helper import *

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
    # TODO: 'embedding' should also be 2

    vocab_size_sub = vocab_size_master

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
        master_helper = ScheduledEmbeddingTrainingHelper(
            inputs,
            sequence_length,
            embedding,
            sampling_probability)

        # TODO: specify this
        sub_helper = ScheduledEmbeddingTrainingHelper(
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
        master_helper = TrainingHelper(inputs, sequence_length)

        # TODO: specify this
        sub_helper = HierarchicalTrainingHelper(inputs, sequence_length)

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

    sub_decoder = BasicSubDecoder(
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
