"""Defines a sequence to sequence model with multiple input features. For
example, this could be words, parts of speech, and lemmas that are embedded in
parallel and concatenated into a single input embedding. The features are
separate data files with separate vocabularies.
"""

import tensorflow as tf
import opennmt as onmt
from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_words_vocabulary",
              embedding_size=None,
              embedding_file_key="words_embedding"),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="feature_vocabulary",
              embedding_size=32)],
          reducer=onmt.layers.ConcatReducer()),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=None,
          embedding_file_key="words_embedding"),
      num_layers=2,
      num_units=256,
      num_heads=4,
      ffn_inner_dim=1024,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1,
      share_embeddings=EmbeddingsSharingLevel.SOURCE_TARGET_INPUT)