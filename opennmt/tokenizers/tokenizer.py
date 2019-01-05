# -*- coding: utf-8 -*-

"""Define base tokenizers."""

import io
import sys
import os
import abc
import copy
import six

import tensorflow as tf
import yaml

from opennmt.utils.misc import print_bytes

def _make_config_asset_file(config, asset_path):
  asset_config = copy.deepcopy(config)
  for key, value in six.iteritems(asset_config):
    # Only keep the basename for files (that should also be registered as assets).
    if isinstance(value, six.string_types) and os.path.exists(value):
      asset_config[key] = os.path.basename(value)
  with open(asset_path, "w") as asset_file:
    yaml.dump(asset_config, stream=asset_file, default_flow_style=False)


@six.add_metaclass(abc.ABCMeta)
class Tokenizer(object):
  """Base class for tokenizers."""

  def __init__(self, configuration_file_or_key=None, params=None):
    """Initializes the tokenizer.

    Args:
      configuration_file_or_key: The YAML configuration file or a the key to
        the YAML configuration file.
    """
    self._configuration_key = None
    if params is not None:
      self._config = params
    else:
      self._config = {}
      if configuration_file_or_key is not None and os.path.isfile(configuration_file_or_key):
        configuration_file = configuration_file_or_key
        with tf.gfile.Open(configuration_file, mode="rb") as conf_file:
          self._config = yaml.load(conf_file)
      else:
        self._configuration_key = configuration_file_or_key

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    """Initializes the tokenizer (e.g. load BPE models).

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    asset_dir: The directory where assets can be written. If ``None``, no
        assets are returned.
      asset_prefix: The prefix to attach to assets filename.
    Returns:
      A dictionary containing additional assets used by the tokenizer.
    """
    assets = {}
    tf.logging.info(" >>>> [tokenizer.py Class Tokenizer initialize]: self._configuration_file_key = {}".format(self._configuration_file_key))
    if self._configuration_key is not None:
      configuration = metadata[self._configuration_key]
      if isinstance(configuration, dict):
        self._config = configuration
      else:
        with tf.gfile.Open(configuration, mode="rb") as conf_file:
          self._config = yaml.load(conf_file)
    if self._config and asset_dir is not None:
      asset_name = "%stokenizer_config.yml" % asset_prefix
      asset_path = os.path.join(asset_dir, asset_name)
      _make_config_asset_file(self._config, asset_path)
      assets[asset_name] = asset_path
    return assets

  def tokenize_stream(self, input_stream=sys.stdin, output_stream=sys.stdout, delimiter=" "):
    """Tokenizes a stream of sentences.

    Args:
      input_stream: The input stream.
      output_stream: The output stream.
      delimiter: The token delimiter to use for text serialization.
    """
    for line in input_stream:
      line = line.strip()
      tokens = self.tokenize(line)
      merged_tokens = delimiter.join(tokens)
      print_bytes(tf.compat.as_bytes(merged_tokens), stream=output_stream)

  def detokenize_stream(self, input_stream=sys.stdin, output_stream=sys.stdout, delimiter=" "):
    """Detokenizes a stream of sentences.

    Args:
      input_stream: The input stream.
      output_stream: The output stream.
      delimiter: The token delimiter used for text serialization.
    """
    for line in input_stream:
      tokens = line.strip().split(delimiter)
      string = self.detokenize(tokens)
      print_bytes(tf.compat.as_bytes(string), stream=output_stream)

  def tokenize(self, text):
    """Tokenizes text.

    Args:
      text: The text to tokenize as a ``tf.Tensor`` or Python string.

    Returns:
      A 1-D string ``tf.Tensor`` if :obj:`text` is a ``tf.Tensor`` or a list of
      Python unicode strings otherwise.

    Raises:
      ValueError: if the rank of :obj:`text` is greater than 0.
    """
    if tf.contrib.framework.is_tensor(text):
      rank = len(text.get_shape().as_list())
      if rank == 0:
        return self._tokenize_tensor(text)
      else:
        raise ValueError("Unsupported tensor rank for tokenization: {}".format(rank))
    else:
      text = tf.compat.as_text(text)
      return self._tokenize_string(text)

  def detokenize(self, tokens, sequence_length=None):
    """Detokenizes tokens.

    The Tensor version supports batches of tokens.

    Args:
      tokens: The tokens as a 1-D or 2-D ``tf.Tensor`` or list of Python
        strings.
      sequence_length: The length of each sequence. Required if :obj:`tokens`
        is a ``tf.Tensor``.

    Returns:
      A 0-D or 1-D string ``tf.Tensor`` if :obj:`tokens` is a ``tf.Tensor`` or a
      Python unicode strings otherwise.

    Raises:
      ValueError: if the rank of :obj:`tokens` is greater than 2.
      ValueError: if :obj:`tokens` is a 2-D ``tf.Tensor`` and
        :obj:`sequence_length` is not set.
    """
    if tf.contrib.framework.is_tensor(tokens):
      rank = len(tokens.get_shape().as_list())
      if rank == 1:
        return self._detokenize_tensor(tokens)
      elif rank == 2:
        if sequence_length is None:
          raise ValueError("sequence_length is required for Tensor detokenization")
        return self._detokenize_batch_tensor(tokens, sequence_length)
      else:
        raise ValueError("Unsupported tensor rank for detokenization: {}".format(rank))
    else:
      tokens = [tf.compat.as_text(token) for token in tokens]
      return self._detokenize_string(tokens)

  def _tokenize_tensor(self, text):
    """Tokenizes a tensor.

    When not overriden, this default implementation uses a ``tf.py_func``
    operation to call the string-based tokenization.

    Args:
      text: A 1-D string ``tf.Tensor``.

    Returns:
      A 1-D string ``tf.Tensor``.
    """
    text = tf.py_func(
        lambda x: tf.compat.as_bytes("\0".join(self.tokenize(x))), [text], tf.string)
    tokens = tf.string_split([text], delimiter="\0").values
    return tokens

  def _detokenize_tensor(self, tokens):
    """Detokenizes tokens.

    When not overriden, this default implementation uses a ``tf.py_func``
    operation to call the string-based detokenization.

    Args:
      tokens: A 1-D ``tf.Tensor``.

    Returns:
      A 0-D string ``tf.Tensor``.
    """
    return tf.py_func(self.detokenize, [tokens], tf.string)

  def _detokenize_batch_tensor(self, tokens, sequence_length):
    """Detokenizes a batch of tokens.

    When not overriden, this default implementation calls _detokenize_tensor on
    each tensor within the batch.

    Args:
      tokens: A 2-D ``tf.Tensor``.

    Returns:
      A 1-D string ``tf.Tensor``.
    """
    return tf.map_fn(
        lambda x: self._detokenize_tensor(x[0][:x[1]]),
        (tokens, sequence_length),
        dtype=tf.string,
        back_prop=False)

  @abc.abstractmethod
  def _tokenize_string(self, text):
    """Tokenizes a Python unicode string.

    This method should be thread-safe.

    Args:
      text: A Python unicode string.

    Returns:
      A list of Python unicode strings.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _detokenize_string(self, tokens):
    """Detokenizes tokens.

    Args:
      tokens: A list of Python unicode strings.

    Returns:
      A unicode Python string.
    """
    raise NotImplementedError()


class SpaceTokenizer(Tokenizer):
  """A tokenizer that splits on spaces."""

  def _tokenize_tensor(self, text):
    tf.logging.info(" >>>> [tokenizer.py Class SpaceTokenizer _tokenize_tensor] text = {}".format(text))
    return tf.string_split([text], delimiter=" ").values

  def _detokenize_tensor(self, tokens):
    return tf.reduce_join(tokens, axis=0, separator=" ")

  def _tokenize_string(self, text):
    return text.split()

  def _detokenize_string(self, tokens):
    return " ".join(tokens)


class CharacterTokenizer(Tokenizer):
  """A tokenizer that splits unicode characters."""

  def _tokenize_string(self, text):
    return list(text.replace(" ", u"▁"))

  def _detokenize_string(self, tokens):
    return "".join(tokens).replace(u"▁", " ")
