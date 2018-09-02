"""Standard sequence-to-sequence model."""

import tensorflow as tf
import sys
import opennmt.constants as constants
import opennmt.inputters as inputters

from opennmt.models.model import Model
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.misc import print_bytes
from opennmt.decoders.decoder import get_sampling_probability


def shift_target_sequence(inputter, data):
  """Prepares shifted target sequences.

  Given a target sequence ``a b c``, the decoder input should be
  ``<s> a b c`` and the output should be ``a b c </s>`` for the dynamic
  decoding to start on ``<s>`` and stop on ``</s>``.

  Args:
    inputter: The :class:`opennmt.inputters.inputter.Inputter` that processed
      :obj:`data`.
    data: A dict of ``tf.Tensor`` containing ``ids`` and ``length`` keys.

  Returns:
    The updated :obj:`data` dictionary with ``ids`` the sequence prefixed
    with the start token id and ``ids_out`` the sequence suffixed with
    the end token id. Additionally, the ``length`` is increased by 1
    to reflect the added token on both sequences.
  """
  tf.logging.info(" >> [sequence_to_sequence.py] shift_target_sequence")
  bos = tf.cast(tf.constant([constants.START_OF_SENTENCE_ID]), tf.int64)
  eos = tf.cast(tf.constant([constants.END_OF_SENTENCE_ID]), tf.int64)

  ids = data["ids"]
  length = data["length"]

  data = inputter.set_data_field(data, "ids_out", tf.concat([ids, eos], axis=0))
  data = inputter.set_data_field(data, "ids", tf.concat([bos, ids], axis=0))

  # Increment length accordingly.
  inputter.set_data_field(data, "length", length + 1)

  return data

# TODO: share embedding is still ugly
def _maybe_reuse_embedding_fn(embedding_fn, scope=None):
  def _scoped_embedding_fn(ids):
    try:
      with tf.variable_scope(scope):
        return embedding_fn(ids)
    except ValueError:
      with tf.variable_scope(scope, reuse=True):
        return embedding_fn(ids)
  return _scoped_embedding_fn

class EmbeddingsSharingLevel(object):
  """Level of embeddings sharing.
   Possible values are:
    * ``NONE``: no sharing (default)
   * ``SOURCE_TARGET_INPUT``: share source and target word embeddings
  """
  NONE = 0
  SOURCE_TARGET_INPUT = 1

class SequenceToSequence(Model):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE,
               daisy_chain_variables=False,
               name="seq2seq"):
    """Initializes a sequence-to-sequence model.

    Args:
      source_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the source data.
      target_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the target data.
                        Currently, only the :class:`opennmt.inputters.text_inputter.WordEmbedder` is supported.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the source.
      decoder: A :class:`opennmt.decoders.decoder.Decoder` to decode the target.
      share_embeddings: Level of embeddings sharing, see
        :class:`opennmt.models.sequence_to_sequence.EmbeddingsSharingLevel`
        for possible values.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain between devices for this model.
                                Not compatible with RNN based models.
      name: The name of this model.

    Raises:
      TypeError: if :obj:`target_inputter` is not a
        :class:`opennmt.inputters.text_inputter.WordEmbedder` (same for
        :obj:`source_inputter` when embeddings sharing is enabled) or if
        :obj:`source_inputter` and :obj:`target_inputter` do not have the same
        ``dtype``.
    """
    if source_inputter.dtype != target_inputter.dtype:
      raise TypeError(
          "Source and target inputters must have the same dtype, "
          "saw: {} and {}".format(source_inputter.dtype, target_inputter.dtype))
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if share_embeddings == EmbeddingsSharingLevel.SOURCE_TARGET_INPUT:
      if not isinstance(source_inputter, inputters.WordEmbedder) and \
              not (isinstance(source_inputter, inputters.ParallelInputter)
                   and isinstance(source_inputter.inputters[0], inputters.WordEmbedder)):
        raise TypeError("Sharing embeddings requires both inputters to be a WordEmbedder or"
                        "the 0th inputter of the ParallelInputter must be a WordEmbedder")

    super(SequenceToSequence, self).__init__(
        name,
        features_inputter=source_inputter,
        labels_inputter=target_inputter,
        daisy_chain_variables=daisy_chain_variables)

    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.source_inputter = source_inputter
    self.target_inputter = target_inputter
    tf.logging.info(" >> [sequence_to_sequence.py __init__] self.target_inputter.add_process_hooks([shift_target_sequence])")
    self.target_inputter.add_process_hooks([shift_target_sequence])
    self.debug = []

  def _get_input_scope(self, default_name=""):
    if self.share_embeddings == EmbeddingsSharingLevel.SOURCE_TARGET_INPUT:
      name = "decoder"
    else:
      name = default_name
    return tf.VariableScope(None, name=tf.get_variable_scope().name + "/" + name)

  def _build(self, features, labels, params, mode, config=None):

      tf.logging.info(" >> [sequence_to_sequence.py _build] mode = <{}> \nfeatures = {}; \nlabels = {}".format(mode, features, labels))

      features_length = self._get_features_length(features)
      log_dir = config.model_dir if config is not None else None

      source_input_scope = self._get_input_scope(default_name="encoder")
      target_input_scope = self._get_input_scope(default_name="decoder")
      tf.logging.info(" >> [sequence_to_sequence.py _build] source_input_scope = %s ;  target_input_scope = %s"%(source_input_scope.name, target_input_scope.name))
      source_inputs = _maybe_reuse_embedding_fn(
          lambda ids: self.source_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
          scope=source_input_scope)(features)

      tf.logging.info(" >> [sequence_to_sequence.py _build] source_inputter = %s"%self.source_inputter)
      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
          encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.encode(
              source_inputs,
              sequence_length=features_length,
              mode=mode)
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_outputs = {}".format(encoder_outputs))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_state (initial_state) = {}".format(encoder_state))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_sequence_length = {}".format(encoder_sequence_length))

      target_vocab_size = self.target_inputter.vocabulary_size
      target_dtype = self.target_inputter.dtype
      target_embedding_fn = _maybe_reuse_embedding_fn(
          lambda ids: self.target_inputter.transform(ids, mode=mode),
          scope=target_input_scope)

      tf.logging.info(" >> [sequence_to_sequence.py _build] target_inputter = %s" % self.target_inputter)
      if labels is not None:
          target_inputs = _maybe_reuse_embedding_fn(
              lambda ids: self.target_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
              scope=target_input_scope)(labels)

          with tf.variable_scope("decoder"):
              sampling_probability = None
              if mode == tf.estimator.ModeKeys.TRAIN:
                  sampling_probability = get_sampling_probability(
                      tf.train.get_or_create_global_step(),
                      read_probability=params.get("scheduled_sampling_read_probability"),
                      schedule_type=params.get("scheduled_sampling_type"),
                      k=params.get("scheduled_sampling_k"))

              logits, rnn_outputs, state, length = self.decoder.decode(
                  target_inputs,
                  self._get_labels_length(labels),
                  vocab_size_master=target_vocab_size,
                  initial_state=encoder_state,
                  sampling_probability=sampling_probability,
                  embedding=target_embedding_fn,
                  mode=mode,
                  memory=encoder_outputs,
                  memory_sequence_length=encoder_sequence_length)
          tf.logging.info(" >> [sequence_to_sequence.py _build] logits = {}".format(logits))
          tf.logging.info(" >> [sequence_to_sequence.py _build] rnn_outputs = {}".format(rnn_outputs))
          tf.logging.info(" >> [sequence_to_sequence.py _build] state = {}".format(state))
          tf.logging.info(" >> [sequence_to_sequence.py _build] length = {}".format(length))
      else:
          logits = None

      if mode != tf.estimator.ModeKeys.TRAIN:
          with tf.variable_scope("decoder", reuse=labels is not None):
              tf.logging.info(" >> [sequence_to_sequence.py _build] mode != tf.estimator.ModeKeys.TRAIN")
              batch_size = tf.shape(encoder_sequence_length)[0]
              beam_width = params.get("beam_width", 1)
              tf.logging.info(" >> [sequence_to_sequence.py _build] beam_width = %d"%beam_width)
              maximum_iterations = params.get("maximum_iterations", 250)
              start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
              end_token = constants.END_OF_SENTENCE_ID

              if beam_width <= 1:
                  tf.logging.info(" >> [sequence_to_sequence.py _build] dynamic_decode ...")
                  sampled_ids, _, sampled_length, log_probs, alignment = self.decoder.dynamic_decode(
                      target_embedding_fn,
                      start_tokens,
                      end_token,
                      vocab_size=target_vocab_size,
                      initial_state=encoder_state,
                      maximum_iterations=maximum_iterations,
                      mode=mode,
                      memory=encoder_outputs,
                      memory_sequence_length=encoder_sequence_length,
                      dtype=target_dtype,
                      return_alignment_history=True)
              else:
                  tf.logging.info(" >> [sequence_to_sequence.py _build] dynamic_decode ...")
                  length_penalty = params.get("length_penalty", 0)
                  sampled_ids, _, sampled_length, log_probs, alignment = (
                      self.decoder.dynamic_decode_and_search(
                          target_embedding_fn,
                          start_tokens,
                          end_token,
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

          target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
              self.target_inputter.vocabulary_file,
              vocab_size=target_vocab_size - self.target_inputter.num_oov_buckets,
              default_value=constants.UNKNOWN_TOKEN)
          tf.logging.info(" >> [sequence_to_sequence.py _build] index_to_string_table_from_file")
          target_tokens = target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
          tf.logging.info(" >> [sequence_to_sequence.py _build] target_vocab_rev.lookup")

          if params.get("replace_unknown_target", False):
              if alignment is None:
                  raise TypeError("replace_unknown_target is not compatible with decoders "
                                  "that don't return alignment history")
              if isinstance(self.source_inputter, inputters.WordEmbedder):
                  source_tokens = features["tokens"]
              elif isinstance(self.source_inputter, inputters.ParallelInputter)\
                      and isinstance(self.source_inputter.inputters[0], inputters.WordEmbedder):
                  source_tokens = features["inputter_0_tokens"]
              else:
                  raise TypeError("replace_unknown_target is only defined when the source "
                                  "inputter is a WordEmbedder or ParallelInputter with the 0th inputter being a WordEmbedder")
              if beam_width > 1:
                  source_tokens = tf.contrib.seq2seq.tile_batch(source_tokens, multiplier=beam_width)
              # Merge batch and beam dimensions.
              original_shape = tf.shape(target_tokens)
              target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
              attention = tf.reshape(alignment, [-1, tf.shape(alignment)[2], tf.shape(alignment)[3]])
              replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
              target_tokens = tf.reshape(replaced_target_tokens, original_shape)

          predictions = {
              "tokens": target_tokens,
              "length": sampled_length,
              "log_probs": log_probs
          }
          if alignment is not None:
              predictions["alignment"] = alignment
      else:
          predictions = None

      return logits, predictions

  def _compute_loss(self, features, labels, outputs, params, mode):
    tf.logging.info(" >> [sequence_to_sequence.py _compute_loss]")
    return cross_entropy_sequence_loss(
        outputs,
        labels["ids_out"],
        self._get_labels_length(labels),
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        mode=mode)

  def print_prediction(self, prediction, params=None, stream=None):
    n_best = params and params.get("n_best")
    n_best = n_best or 1

    if n_best > len(prediction["tokens"]):
      raise ValueError("n_best cannot be greater than beam_width")

    for i in range(n_best):
      tokens = prediction["tokens"][i][:prediction["length"][i] - 1] # Ignore </s>.
      sentence = self.target_inputter.tokenizer.detokenize(tokens)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)


def align_tokens_from_attention(tokens, attention):
  """Returns aligned tokens from the attention.

  Args:
    tokens: The tokens on which the attention is applied as a string
      ``tf.Tensor`` of shape :math:`[B, T_s]`.
    attention: The attention vector of shape :math:`[B, T_t, T_s]`.

  Returns:
    The aligned tokens as a string ``tf.Tensor`` of shape :math:`[B, T_t]`.
  """
  alignment = tf.argmax(attention, axis=-1, output_type=tf.int32)
  batch_size = tf.shape(tokens)[0]
  max_time = tf.shape(attention)[1]
  batch_ids = tf.range(batch_size)
  batch_ids = tf.tile(batch_ids, [max_time])
  batch_ids = tf.reshape(batch_ids, [max_time, batch_size])
  batch_ids = tf.transpose(batch_ids, perm=[1, 0])
  aligned_pos = tf.stack([batch_ids, alignment], axis=-1)
  aligned_tokens = tf.gather_nd(tokens, aligned_pos)
  return aligned_tokens

def replace_unknown_target(target_tokens,
                           source_tokens,
                           attention,
                           unknown_token=constants.UNKNOWN_TOKEN):
  """Replaces all target unknown tokens by the source token with the highest
  attention.

  Args:
    target_tokens: A a string ``tf.Tensor`` of shape :math:`[B, T_t]`.
    source_tokens: A a string ``tf.Tensor`` of shape :math:`[B, T_s]`.
    attention: The attention vector of shape :math:`[B, T_t, T_s]`.
    unknown_token: The target token to replace.

  Returns:
    A string ``tf.Tensor`` with the same shape and type as :obj:`target_tokens`
    but will all instances of :obj:`unknown_token` replaced by the aligned source
    token.
  """
  aligned_source_tokens = align_tokens_from_attention(source_tokens, attention)
  return tf.where(
      tf.equal(target_tokens, unknown_token),
      x=aligned_source_tokens,
      y=target_tokens)
