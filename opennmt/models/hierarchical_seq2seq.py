"""Standard sequence-to-sequence model."""

import tensorflow as tf
import sys
import opennmt.constants as constants
import opennmt.inputters as inputters

from opennmt.models.model import Model
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.misc import print_bytes
from opennmt.decoders.decoder import get_sampling_probability
from opennmt.utils.hooks import add_counter
from opennmt.models.sequence_to_sequence import _maybe_reuse_embedding_fn, EmbeddingsSharingLevel, shift_target_sequence
from opennmt.utils.misc import add_dict_to_collection
from opennmt.layers.reducer import align_in_time, align_in_master_time_nodepth

log_separator = "\nINFO:tensorflow:{}\n".format("*"*50)

def shift_target_sequence_v2(inputter, data):
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
  tf.logging.info(" >> [hierarchical_seq2seq.py] shift_target_sequence_v2")

  ids = data["ids"]
  length = data["length"]

  ids_out = ids[1:]
  ids_in = ids[:-1]

  data = inputter.set_data_field(data, "ids_out", ids_out)
  data = inputter.set_data_field(data, "ids", ids_in)

  # decrement length accordingly.
  inputter.set_data_field(data, "length", length - 1)

  return data


class HierarchicalSequenceToSequence(Model):
  """A hierarchical sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               sub_target_inputter,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE,
               daisy_chain_variables=False,
               name="seq2seq",
               shifted=None):
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

    if share_embeddings == EmbeddingsSharingLevel.SOURCE_TARGET_INPUT:
      if not isinstance(source_inputter, inputters.WordEmbedder) and \
              not (isinstance(source_inputter, inputters.ParallelInputter)
                   and isinstance(source_inputter.inputters[0], inputters.WordEmbedder)):
        raise TypeError("Sharing embeddings requires both inputters to be a WordEmbedder or"
                        "the 0th inputter of the ParallelInputter must be a WordEmbedder")

    super(HierarchicalSequenceToSequence, self).__init__(
        name,
        features_inputter=source_inputter,
        labels_inputter=target_inputter,
        daisy_chain_variables=daisy_chain_variables)

    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.source_inputter = source_inputter
    self.target_inputter = target_inputter
    self.sub_target_inputter = sub_target_inputter
    self.debug = []
    self.shifted = shifted

    tf.logging.info(" >> [hierarchical_seq2seq.py __init__] self.target_inputter.add_process_hooks([shift_target_sequence])")
    shift_target_fn = shift_target_sequence_v2 if self.shifted is not None else shift_target_sequence
    self.target_inputter.add_process_hooks([shift_target_sequence])
    self.sub_target_inputter.add_process_hooks([shift_target_fn])

  def _get_input_scope(self, default_name=""):
    if self.share_embeddings == EmbeddingsSharingLevel.SOURCE_TARGET_INPUT:
      name = "shared_embeddings"
    elif self.share_embeddings == EmbeddingsSharingLevel.SOURCE_CONTROLLER_INPUT and isinstance(default_name, tuple):
      name = "partially_shared_embeddings"
      return tuple([tf.VariableScope(None, name=tf.get_variable_scope().name + "/" + name) for i in range(len(default_name))])
    else:
      name = default_name
    return tf.VariableScope(None, name=tf.get_variable_scope().name + "/" + name)

  def _build(self, features, labels, params, mode, config=None):

      tf.logging.info(log_separator+" >> [hierarchical_seq2seq.py _build] mode = {}".format(mode))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] features = \n{}".format(features))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] len(features) = {}\n".format(len(features)))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] labels = \n{}".format(labels))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] len(labels) = {}".format(len(labels) if labels is not None else None))

      features_length = self._get_features_length(features)
      log_dir = config.model_dir if config is not None else None

      if self.share_embeddings == EmbeddingsSharingLevel.SOURCE_CONTROLLER_INPUT:
        source_input_scope, target_input_scope = self._get_input_scope(default_name=("encoder", "controller"))
        sub_target_input_scope = self._get_input_scope(default_name="fragment_decoder")
      else:
        source_input_scope = self._get_input_scope(default_name="encoder")
        target_input_scope = self._get_input_scope(default_name="decoder")
        sub_target_input_scope = target_input_scope

      tf.logging.info(" >> [hierarchical_seq2seq.py _build] source_input_scope = {}".format(source_input_scope.name))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] target_input_scope = {}".format(target_input_scope.name))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] sub_target_input_scope = {}".format(sub_target_input_scope.name))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] source_inputter = {}".format(self.source_inputter))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] target_inputter = {}".format(self.target_inputter))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] sub_target_inputter = {}".format(self.sub_target_inputter))

      tf.logging.info(log_separator+" >> [hierarchical_seq2seq.py _build] self.source_inputter.transform_data")
      source_inputs = _maybe_reuse_embedding_fn(
          lambda ids: self.source_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
          scope=source_input_scope)(features)
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] source_inputs = {}".format(source_inputs))

      with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
          encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.encode(
              source_inputs,
              sequence_length=features_length,
              mode=mode)
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_outputs = {}".format(encoder_outputs))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_state (initial_state) = {}".format(encoder_state))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_sequence_length = {}".format(encoder_sequence_length))

      master_target_vocab_size = self.target_inputter.get_vocab_size()
      sub_target_vocab_size = self.sub_target_inputter.get_vocab_size()
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_sequence_length = {}".format(master_target_vocab_size))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] encoder_sequence_length = {}".format(sub_target_vocab_size))

      target_dtype = self.target_inputter.dtype

      target_embedding_fn = _maybe_reuse_embedding_fn(
          lambda ids: self.target_inputter.transform(ids, mode=mode),
          scope=target_input_scope) # callable

      sub_target_embedding_fn = _maybe_reuse_embedding_fn(
          lambda ids: self.sub_target_inputter.transform(ids, mode=mode),
          scope=sub_target_input_scope) # callable

      tf.logging.info(" >> [hierarchical_seq2seq.py _build] target_embedding_fn = {}".format(target_embedding_fn))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] sub_target_embedding_fn = {}".format(sub_target_embedding_fn))
      tf.logging.info(" >> [hierarchical_seq2seq.py _build] target_inputter = {}".format(self.target_inputter))
      if labels is not None:
          master_labels, sub_labels = labels
          tf.logging.info(log_separator + " >> [hierarchical_seq2seq.py _build] len(master_labels) = {}".format(len(master_labels)))
          tf.logging.info(log_separator + " >> [hierarchical_seq2seq.py _build] master_labels = {}".format(master_labels))
          tf.logging.info(log_separator + " >> [hierarchical_seq2seq.py _build] len(sub_labels) = {}".format(len(sub_labels)))
          tf.logging.info(log_separator + " >> [hierarchical_seq2seq.py _build] sub_labels = {}".format(sub_labels))

          target_inputs = _maybe_reuse_embedding_fn(
              lambda ids: self.target_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
              scope=target_input_scope)(master_labels)
          tf.logging.info(log_separator + " >> [hierarchical_seq2seq.py _build] target_inputs = {}".format(target_inputs))

          sub_target_inputs = _maybe_reuse_embedding_fn(
              lambda ids: self.sub_target_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
              scope=sub_target_input_scope)(sub_labels)
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] sub_target_inputs = {}".format(sub_target_inputs))
          tf.logging.info(log_separator + " >> [hierarchical_seq2seq.py _build] self.decoder.decode ...")

          with tf.variable_scope("decoder"):
              sampling_probability = None
              if mode == tf.estimator.ModeKeys.TRAIN:
                  sampling_probability = get_sampling_probability(
                      tf.train.get_or_create_global_step(),
                      read_probability=params.get("scheduled_sampling_read_probability"),
                      schedule_type=params.get("scheduled_sampling_type"),
                      k=params.get("scheduled_sampling_k"))

              logits, logits_sub, state, length, sequence_mask_sub = self.decoder.decode(
                  (target_inputs, sub_target_inputs),
                  self._get_labels_length(labels, to_reduce=True),
                  vocab_size=(master_target_vocab_size, sub_target_vocab_size),
                  initial_state=encoder_state,
                  sampling_probability=sampling_probability,
                  embedding=(target_embedding_fn, sub_target_embedding_fn),
                  mode=mode,
                  memory=encoder_outputs,
                  memory_sequence_length=encoder_sequence_length,
                  shifted=self.shifted)
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] logits = {}".format(logits))
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] logits_sub = {}".format(logits_sub))
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] state = {}".format(state))
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] length = {}".format(length))
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] sequence_mask_sub = {}".format(sequence_mask_sub))
          add_dict_to_collection("debug", {"decoder_logits": tf.shape(logits),
                                           "decoder_target_inputs": tf.shape(target_inputs),
                                           "decoder_logits_sub": tf.shape(logits_sub),
                                           "decoder_length": tf.shape(length),
                                           "sequence_mask_sub": tf.shape(sequence_mask_sub)
                                           })
      else:
          logits = None
          logits_sub = None
          sequence_mask_sub = None

      if mode != tf.estimator.ModeKeys.TRAIN:
          with tf.variable_scope("decoder", reuse=labels is not None):
              tf.logging.info(" >> [hierarchical_seq2seq.py _build] mode != tf.estimator.ModeKeys.TRAIN")
              batch_size = tf.shape(encoder_sequence_length)[0]
              beam_width = params.get("beam_width", 1)
              tf.logging.info(" >> [hierarchical_seq2seq.py _build] beam_width = %d"%beam_width)
              maximum_iterations = params.get("maximum_iterations", 0)
              tf.logging.info(" >> [hierarchical_seq2seq.py _build] maximum_iterations = {}".format(maximum_iterations))
              assert maximum_iterations == self.sub_target_inputter.num + 1
              sub_maximum_iterations = params.get("sub_maximum_iterations", 50)
              start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
              end_token = constants.END_OF_SENTENCE_ID

              if beam_width <= 1:
                  tf.logging.info(" >> [hierarchical_seq2seq.py _build] dynamic_decode ...")
                  sampled_ids, _, sampled_length, log_probs, alignment = self.decoder.dynamic_decode(
                      embedding=(target_embedding_fn, sub_target_embedding_fn),
                      start_tokens=start_tokens,
                      end_token=end_token,
                      vocab_size=(master_target_vocab_size, sub_target_vocab_size),
                      initial_state=encoder_state,
                      maximum_iterations=maximum_iterations,
                      sub_maximum_iterations=sub_maximum_iterations,
                      mode=mode,
                      memory=encoder_outputs,
                      memory_sequence_length=encoder_sequence_length,
                      dtype=target_dtype,
                      return_alignment_history=True,
                      shifted=self.shifted)
                  tf.logging.info(" >> [hierarchical_seq2seq.py _build] sampled_ids = {}".format(sampled_ids))
                  tf.logging.info(" >> [hierarchical_seq2seq.py _build] sampled_length = {}".format(sampled_length))
                  tf.logging.info(" >> [hierarchical_seq2seq.py _build] log_probs = {}".format(log_probs))
                  tf.logging.info(" >> [hierarchical_seq2seq.py _build] alignment = {}".format(alignment))

              else:
                  tf.logging.info(" >> [hierarchical_seq2seq.py _build] dynamic_decode_and_search ...")
                  length_penalty = params.get("length_penalty", 0)
                  sampled_ids, _, sampled_length, log_probs, alignment = (
                      self.decoder.dynamic_decode_and_search(
                          target_embedding_fn,
                          start_tokens,
                          end_token,
                          vocab_size=master_target_vocab_size,
                          initial_state=encoder_state,
                          beam_width=beam_width,
                          length_penalty=length_penalty,
                          maximum_iterations=maximum_iterations,
                          mode=mode,
                          memory=encoder_outputs,
                          memory_sequence_length=encoder_sequence_length,
                          dtype=target_dtype,
                          return_alignment_history=True,
                          shifted=self.shifted))

          target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
              self.target_inputter.vocabulary_file,
              vocab_size=master_target_vocab_size - self.target_inputter.num_oov_buckets,
              default_value=constants.UNKNOWN_TOKEN)

          sub_target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
              self.sub_target_inputter.vocabulary_file,
              vocab_size=sub_target_vocab_size - self.sub_target_inputter.num_oov_buckets,
              default_value=constants.UNKNOWN_TOKEN)

          tf.logging.info(" >> [hierarchical_seq2seq.py _build] index_to_string_table_from_file")
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] sampled_ids = {}".format(sampled_ids))
          if isinstance(sampled_ids, tuple):
            assert isinstance(sampled_length, tuple)
            # TODO: this is ugly
            sampled_length, sub_sampled_length = sampled_length
            master_ids, sub_ids = sampled_ids
            add_dict_to_collection("debug", {"sampled_length": tf.shape(sampled_length),
                                             "sub_sampled_length": tf.shape(sub_sampled_length),
                                             "master_ids": tf.shape(master_ids),
                                             "sub_ids": tf.shape(sub_ids)
                                             })

            target_tokens = target_vocab_rev.lookup(tf.cast(master_ids, tf.int64))
            target_tokens_sub = sub_target_vocab_rev.lookup(tf.cast(sub_ids, tf.int64))
          else:
            target_tokens = target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))
            target_tokens_sub = None
            sub_sampled_length = None

          tf.logging.info(" >> [hierarchical_seq2seq.py _build] target_vocab_rev.lookup")
          tf.logging.info(" >> [hierarchical_seq2seq.py _build] target_tokens_sub = {}".format(target_tokens_sub))

          # NOTE: replace_unknown_target is only for controller now
          if params.get("replace_unknown_target", False):
              tf.logging.info(" >> [hierarchical_seq2seq.py _build] replace_unknown_target ... ")
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
          if target_tokens_sub is not None:
              predictions["tokens_sub"] = target_tokens_sub
          if sub_sampled_length is not None:
              predictions["length_sub"] = sub_sampled_length
          if alignment is not None:
              predictions["alignment"] = alignment
      else:
          predictions = None

      return (logits, logits_sub, sequence_mask_sub), predictions

  def _initialize(self, metadata):
    """Runs model specific initialization (e.g. vocabularies loading).

    Args:
      metadata: A dictionary containing additional metadata set by the user.
    """
    super(HierarchicalSequenceToSequence, self)._initialize(metadata)
    tf.logging.info(" >> [hierarchical_seq2seq.py _initialize] Initializing with metadata ... ")
    if self.sub_target_inputter is not None:
      tf.logging.info(" >> [hierarchical_seq2seq.py _initialize] self.labels_inputter.initialize(metadata) --- sub_target_inputter = {}".format(self.sub_target_inputter))
      self.sub_target_inputter.initialize(metadata)

  def _get_labels_builder(self, labels_file):
      """Returns the recipe to build labels.

      Args:
        labels_file: The file of labels.

      Returns:
        A tuple ``(tf.data.Dataset, process_fn)``.
      """
      '''
      labels_inputter=target_inputter
      '''
      master_labels_file, sub_labels_file = labels_file
      tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_builder] master_labels_file = {}".format(master_labels_file))
      tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_builder] sub_labels_file = {}".format(sub_labels_file))

      if self.labels_inputter is None:
          raise NotImplementedError()
      tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_builder] self.labels_inputter.make_dataset(master_labels_file)")
      master_labels_dataset = self.labels_inputter.make_dataset(master_labels_file)
      tf.logging.info(" >> [hierarchical_seq2seq.py _input_fn_impl] master_labels_dataset = {} ".format(master_labels_dataset))
      master_labels_process_fn = self.labels_inputter.process
      tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_builder] master_labels_process_fn = {}".format(master_labels_process_fn))

      if self.sub_target_inputter is None:
          raise NotImplementedError()
      tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_builder] self.sub_target_inputter.make_dataset(sub_labels_file)")
      sub_labels_dataset = self.sub_target_inputter.make_dataset(sub_labels_file)
      tf.logging.info(" >> [hierarchical_seq2seq.py _input_fn_impl] sub_labels_dataset = {} ".format(sub_labels_dataset))
      sub_labels_process_fn = self.sub_target_inputter.process
      tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_builder] sub_labels_process_fn = {}".format(sub_labels_process_fn))

      labels_dataset = tf.data.Dataset.zip((master_labels_dataset, sub_labels_dataset))
      process_fn = lambda labels: (master_labels_process_fn(labels[0]), sub_labels_process_fn(labels[1]))

      return labels_dataset, process_fn

  def _get_labels_length(self, labels, to_reduce=False):
    """Returns the labels length.

    Args:
      labels: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor``  or ``None`` if length is undefined.
    """
    tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_length] labels = {}".format(labels))
    tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_length] len(labels) = {}".format(len(labels)))
    master_labels, sub_labels = labels
    tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_length] master_labels = {}".format(master_labels))
    tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_length] sub_labels = {}".format(sub_labels))

    master_length = self.labels_inputter.get_length(master_labels)
    tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_length] master_length = {}".format(master_length))
    sub_length = self.sub_target_inputter.get_length(sub_labels, to_reduce=to_reduce)
    tf.logging.info(" >> [hierarchical_seq2seq.py _get_labels_length] sub_length = {}".format(sub_length))
    if self.labels_inputter is None or self.sub_target_inputter is None:
      return None
    return [master_length, sub_length]

  def _register_word_counters(self, features, labels):
    """Creates word counters for sequences (if any) of :obj:`features` and
    :obj:`labels`.
    """
    tf.logging.info(" >> [hierarchical_seq2seq.py _register_word_counters]")
    features_length = self._get_features_length(features)
    master_length, sub_length = self._get_labels_length(labels)

    with tf.variable_scope("words_per_sec"):
      if features_length is not None:
        add_counter("features", tf.reduce_sum(features_length))
      if master_length is not None:
        add_counter("master_labels", tf.reduce_sum(master_length))
      if sub_length is not None:
        add_counter("sub_labels", tf.reduce_sum(sub_length))

  def _compute_loss_impl(self, logits, labels, sequence_length, params, mode, master_mask=None):
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss_impl] labels = {}".format(labels))
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss_impl] labels[\"ids_out\"] = {}".format(labels["ids_out"]))
    return cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        sequence_length,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        mode=mode,
        master_mask=master_mask)

  def _compute_loss(self, features, labels, outputs, params, mode):

    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] outputs = {}".format(outputs))
    master_labels, sub_labels = labels
    master_logits, sub_logits, sequence_mask_sub = outputs
    master_length, sub_length = self._get_labels_length(labels, to_reduce=True)
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] master_length = {}".format(master_length))
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] sub_length = {}".format(sub_length))
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] \nmaster_labels : \n{}".format("\n".join(["{}".format(x) for x in master_labels.items()])))
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] BEFORE \nsub_labels : \n{}".format("\n".join(["{}".format(x) for x in sub_labels.items()])))

    master_loss = self._compute_loss_impl(master_logits, master_labels, master_length, params, mode)
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] master_loss = {}".format(master_loss))

    '''
     master_length was added by one due to shift_target_sequence, so -1
    '''
    master_mask = tf.sequence_mask(master_length-1, maxlen=tf.shape(master_logits)[1], dtype=tf.float32)
    master_mask = tf.expand_dims(master_mask, axis=-1)
    master_mask = align_in_time(master_mask, tf.shape(sub_logits)[1])
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] master_mask = {}".format(master_mask))
    sub_length = align_in_master_time_nodepth(sub_length, tf.shape(sub_logits)[1])

    sub_labels = self.sub_target_inputter._transform_sub_labels(sub_labels)
    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] AFTER \nsub_labels : \n{}".format("\n".join(["{}".format(x) for x in sub_labels.items()])))
    sub_loss = self._compute_loss_impl(sub_logits, sub_labels, sub_length, params, mode, master_mask=master_mask)
    sub_loss_value, sub_loss_normalizer1, sub_loss_normalizer2 = sub_loss
    tf.summary.scalar("sub_loss_value)", sub_loss_value)
    tf.summary.scalar("sub_loss_normalizer1)", sub_loss_normalizer1)
    tf.summary.scalar("sub_loss_normalizer2)", sub_loss_normalizer2)

    add_dict_to_collection("debug", {"master_length": master_length,
                                     "sub_length": sub_length,
                                     "master_logits": tf.shape(master_logits),
                                     "sub_logits": tf.shape(sub_logits),
                                     "sequence_mask_sub_shape": tf.shape(sequence_mask_sub),
                                     "sequence_mask_sub": sequence_mask_sub,
                                     "sub_loss_value": sub_loss_value,
                                     "sub_loss_normalizer1": sub_loss_normalizer1,
                                     "sub_loss_normalizer2": sub_loss_normalizer2,
                                     })

    tf.logging.info(" >> [hierarchical_seq2seq.py _compute_loss] sub_loss = {}".format(sub_loss))

    return (master_loss, sub_loss)

  def print_prediction(self, prediction, params=None, stream=None, sub_stream=None):
    n_best = params and params.get("n_best")
    n_best = n_best or 1

    # tf.logging.info(" >> [hierarchical_seq2seq.py print_prediction] prediction = {}".format(prediction.keys()))

    if n_best > len(prediction["tokens"]):
      raise ValueError("n_best cannot be greater than beam_width")

    for i in range(n_best):
      tokens = prediction["tokens"][i][:prediction["length"][i]-1] # Ignore </s>.
      sentence = self.target_inputter.tokenizer.detokenize(tokens)
      if params is not None and params.get("with_scores"):
        sentence = "%f ||| %s" % (
            prediction["log_probs"][i] / prediction["length"][i], sentence)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)
      if sub_stream is not None:
        # sub_tokens = prediction["tokens_sub"][i][:prediction["length_sub"][i]]
        sub_tokens = prediction["tokens_sub"][i]
        sub_sentence = self.target_inputter.tokenizer.detokenize(sub_tokens)
        sub_sentence = sub_sentence.replace(" <blank>", "")
        if params is not None and params.get("with_scores"):
            sub_sentence = "%f ||| %s" % (
                prediction["log_probs"][i] / prediction["length_sub"][i], sub_sentence) # log_probs = log_probs_master + log_probs_sub
        print_bytes(tf.compat.as_bytes(sub_sentence), stream=sub_stream)

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

