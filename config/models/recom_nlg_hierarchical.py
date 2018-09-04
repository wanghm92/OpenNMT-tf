"""Defines a sequence to sequence model with multiple input features. For
example, this could be words, parts of speech, and lemmas that are embedded in
parallel and concatenated into a single input embedding. The features are
separate data files with separate vocabularies.
"""

import tensorflow as tf
import opennmt as onmt
from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel

def model():
  return onmt.models.HierarchicalSequenceToSequence(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_words_vocabulary",
              embedding_size=None,
              embedding_file_key="words_embedding"),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="feature_vocabulary",
              embedding_size=64)],
          reducer=onmt.layers.ConcatReducer()),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="master_target_words_vocabulary",
          embedding_size=None,
          embedding_file_key="words_embedding"),
      sub_target_inputter=onmt.inputters.HierarchicalInputter(
          inputter_type=onmt.inputters.WordEmbedder,
          inputter_args=["sub_target_words_vocabulary", None, "words_embedding"],
          num=5),
      encoder=onmt.encoders.BidirectionalRNNEncoder(
          num_layers=1,
          num_units=128,
          reducer=onmt.layers.ConcatReducer(),
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False),
      decoder=onmt.decoders.HierarchicalAttentionalRNNDecoder(
          num_layers=1,
          num_units=128,
          bridge=onmt.layers.CopyBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False),
      share_embeddings=EmbeddingsSharingLevel.SOURCE_TARGET_INPUT)
