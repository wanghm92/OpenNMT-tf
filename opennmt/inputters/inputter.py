"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.layers.reducer import ConcatReducer, PackReducer
from opennmt.utils.misc import extract_prefixed_keys

debug_cnt=0

@six.add_metaclass(abc.ABCMeta)
class Inputter(object):
  """Base class for inputters."""

  def __init__(self, dtype=tf.float32):
    self.volatile = set()
    self.process_hooks = []
    self.dtype = dtype

  def add_process_hooks(self, hooks):
    """Adds processing hooks.

    Processing hooks are additional and model specific data processing
    functions applied after calling this inputter
    :meth:`opennmt.inputters.inputter.Inputter.process` function.

    Args:
      hooks: A list of callables with the signature
        ``(inputter, data) -> data``.
    """
    self.process_hooks.extend(hooks)

  def set_data_field(self, data, key, value, volatile=False):
    """Sets a data field.

    Args:
      data: The data dictionary.
      key: The value key.
      value: The value to assign.
      volatile: If ``True``, the key/value pair will be removed once the
        processing done.

    Returns:
      The updated data dictionary.
    """
    data[key] = value
    if volatile:
      self.volatile.add(key)
    return data

  def remove_data_field(self, data, key):
    """Removes a data field.

    Args:
      data: The data dictionary.
      key: The value key.

    Returns:
      The updated data dictionary.
    """
    del data[key]
    return data

  def get_length(self, unused_data):
    """Returns the length of the input data, if defined."""
    return None

  @abc.abstractmethod
  def make_dataset(self, data_file):
    """Creates the dataset required by this inputter.

    Args:
      data_file: The data file.

    Returns:
      A ``tf.data.Dataset``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    """Returns the size of the dataset.

    Args:
      data_file: The data file.

    Returns:
      The total size.
    """
    raise NotImplementedError()

  def get_serving_input_receiver(self):
    """Returns a serving input receiver for this inputter.

    Returns:
      A ``tf.estimator.export.ServingInputReceiver``.
    """
    receiver_tensors, features = self._get_serving_input()
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def _get_serving_input(self):
    """Returns the input receiver for serving.

    Returns:
      A tuple ``(receiver_tensors, features)`` as described in
      ``tf.estimator.export.ServingInputReceiver``.
    """
    raise NotImplementedError()

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    """Initializes the inputter within the current graph.

    For example, one can create lookup tables in this method
    for their initializer to be added to the current graph
    ``TABLE_INITIALIZERS`` collection.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
      asset_dir: The directory where assets can be written. If ``None``, no
        assets are returned.
      asset_prefix: The prefix to attach to assets filename.
     Returns:
      A dictionary containing additional assets used by the inputter.
    """
    _ = metadata
    _ = asset_dir
    _ = asset_prefix
    return {}

  def process(self, data):
    """Prepares raw data.

    Args:
      data: The raw data.

    Returns:
      A dictionary of ``tf.Tensor``.

    See Also:
      :meth:`opennmt.inputters.inputter.Inputter.transform_data`
    """
    global debug_cnt
    debug_cnt += 1
    tf.logging.info(" >> [inputter.py class Inputter process] data = self._process(data) debug_cnt = {}".format(debug_cnt))
    data = self._process(data)
    tf.logging.info(" >> [inputter.py class Inputter process] after data = self._process(data)\ndata : \n{}".format("\n".join(["{}".format(x) for x in data.items()])))
    tf.logging.info(" >> [inputter.py class Inputter process] applying process_hooks ...")
    for hook in self.process_hooks:
      data = hook(self, data)
    tf.logging.info(" >> [inputter.py class Inputter process] after applying hooks\ndata : \n{}".format("\n".join(["{}".format(x) for x in data.items()])))
    for key in self.volatile:
      data = self.remove_data_field(data, key)
    self.volatile.clear()
    tf.logging.info(" >> [inputter.py class Inputter process] after remove_data_field()\ndata : \n{}".format("\n".join(["{}".format(x) for x in data.items()])))
    return data

  def _process(self, data):
    """Prepares raw data (implementation).

    Subclasses should extend this function to prepare the raw value read
    from the dataset to something they can transform (e.g. processing a
    line of text to a sequence of ids).

    This base implementation makes sure the data is a dictionary so subclasses
    can populate it.

    Args:
      data: The raw data or a dictionary containing the ``raw`` key.

    Returns:
      A dictionary of ``tf.Tensor``.

    Raises:
      ValueError: if :obj:`data` is a dictionary but does not contain the
        ``raw`` key.
    """
    tf.logging.info(" >> [inputter.py class Inputter _process] data = {}".format(data))
    if not isinstance(data, dict):
      tf.logging.info(" >> [inputter.py class Inputter _process] data = self.set_data_field(\"raw\")")
      data = self.set_data_field({}, "raw", data, volatile=True)
    elif "raw" not in data:
      raise ValueError("data must contain the raw dataset value")
    tf.logging.info(" >> [inputter.py class Inputter _process] return data = {}".format(data))
    return data

  def visualize(self, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      log_dir: The active log directory.
    """
    pass

  def transform_data(self, data, mode=tf.estimator.ModeKeys.TRAIN, log_dir=None):
    """Transforms the processed data to an input.

    This is usually a simple forward of a :obj:`data` field to
    :meth:`opennmt.inputters.inputter.Inputter.transform`.

    See also `process`.

    Args:
      data: A dictionary of data fields.
      mode: A ``tf.estimator.ModeKeys`` mode.
      log_dir: The log directory. If set, visualization will be setup.

    Returns:
      The transformed input.
    """
    tf.logging.info(" >> [inputter.py class Inputter transform_data] Embedding Lookup ...")
    inputs = self._transform_data(data, mode)
    if log_dir:
      self.visualize(log_dir)
    return inputs

  @abc.abstractmethod
  def _transform_data(self, data, mode):
    """Implementation of ``transform_data``."""
    raise NotImplementedError()

  @abc.abstractmethod
  def transform(self, inputs, mode):
    """Transforms inputs.

    Args:
      inputs: A (possible nested structure of) ``tf.Tensor`` which depends on
        the inputter.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      The transformed input.
    """
    raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class MultiInputter(Inputter):
  """An inputter that gathers multiple inputters."""

  def __init__(self, inputters):
    if not isinstance(inputters, list) or not inputters:
      raise ValueError("inputters must be a non empty list")
    dtype = inputters[0].dtype
    for inputter in inputters:
      if inputter.dtype != dtype:
        raise TypeError("All inputters must have the same dtype")
    super(MultiInputter, self).__init__(dtype=dtype)
    self.inputters = inputters

  @abc.abstractmethod
  def make_dataset(self, data_file):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    raise NotImplementedError()

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    assets = {}
    tf.logging.info(" >>>> [inputter.py Class MultiInputter initialize]")
    for idx, inputter in enumerate(self.inputters):
      tf.logging.info(" >>>> [inputter.py Class MultiInputter initialize] initializing inputter ---- {} ---- ({}) ".format(idx, inputter))
      assets.update(inputter.initialize(metadata), asset_dir=asset_dir, asset_prefix="%s%d_" % (asset_prefix, i))
    return assets

  def visualize(self, log_dir):
    tf.logging.info(" >>>> [inputter.py Class MultiInputter visualize]")
    self.inputters[0].visualize(log_dir)
    for i, inputter in enumerate(self.inputters[1:]):
        with tf.variable_scope("inputter_{}".format(i+1)):
          inputter.visualize(log_dir)

  @abc.abstractmethod
  def _get_serving_input(self):
    raise NotImplementedError()

  def transform(self, inputs, mode):
    transformed = []
    tf.logging.info(" >>>> [inputter.py Class MultiInputter transform]")
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        transformed.append(inputter.transform(inputs[i], mode))
    return transformed


class ParallelInputter(MultiInputter):
  """An multi inputter that process parallel data."""

  def __init__(self, inputters, reducer=None):
    """Initializes a parallel inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs. If
        set, parallel inputs are assumed to have the same length.
    """
    super(ParallelInputter, self).__init__(inputters)
    self.reducer = reducer

  def get_length(self, data):
    tf.logging.info(" >> [inputter.py class ParallelInputter get_length]")
    lengths = []
    for i, inputter in enumerate(self.inputters):
      sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
      lengths.append(inputter.get_length(sub_data))
    if self.reducer is None:
      return lengths
    else:
      return lengths[0]

  def make_dataset(self, data_file):
    tf.logging.info(" >> [inputter.py class ParallelInputter make_dataset] data_file = {}".format(data_file))
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    datasets = [
        inputter.make_dataset(data)
        for inputter, data in zip(self.inputters, data_file)]
    return tf.data.Dataset.zip(tuple(datasets))

  def get_dataset_size(self, data_file):
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    dataset_sizes = [
        inputter.get_dataset_size(data)
        for inputter, data in zip(self.inputters, data_file)]
    dataset_size = dataset_sizes[0]
    for size in dataset_sizes:
      if size != dataset_size:
        raise RuntimeError("The parallel data files do not have the same size")
    return dataset_size

  def _get_serving_input(self):
    all_receiver_tensors = {}
    all_features = {}
    for i, inputter in enumerate(self.inputters):
      receiver_tensors, features = inputter._get_serving_input()  # pylint: disable=protected-access
      for key, value in six.iteritems(receiver_tensors):
        all_receiver_tensors["{}_{}".format(key, i)] = value
      for key, value in six.iteritems(features):
        all_features["inputter_{}_{}".format(i, key)] = value
    return all_receiver_tensors, all_features

  def _process(self, data):
    tf.logging.info(" >> [inputter.py class ParallelInputter _process] data = {}".format(data))
    processed_data = {}
    for i, inputter in enumerate(self.inputters):
      sub_data = inputter._process(data[i])  # pylint: disable=protected-access
      tf.logging.info(" >> [inputter.py class ParallelInputter _process] sub_data : \n{}".format("\n".join(["{}".format(x) for x in sub_data.items()])))
      tf.logging.info(" >> [inputter.py class ParallelInputter _process] inputter.volatile = {}".format(inputter.volatile))
      for key, value in six.iteritems(sub_data):
        prefixed_key = "inputter_{}_{}".format(i, key)
        processed_data = self.set_data_field(
            processed_data,
            prefixed_key,
            value,
            volatile=key in inputter.volatile)
      tf.logging.info(" >> [inputter.py class ParallelInputter _process] self.volatile = {}".format(self.volatile))
    return processed_data

  def _transform_data(self, data, mode):
    tf.logging.info(" >> [inputter.py class ParallelInputter _transform_data] for i, inputter in enumerate(self.inputters) ...")
    tf.logging.info(" >> [inputter.py class ParallelInputter _transform_data] data = {}".format(data))
    transformed = []

    inputter = self.inputters[0]
    tf.logging.info(" >> [inputter.py class ParallelInputter _transform_data] self.inputters[0] = {}".format(inputter))
    sub_data = extract_prefixed_keys(data, "inputter_{}_".format(0))
    tf.logging.info(" >> [inputter.py class ParallelInputter _transform_data] sub_data = {}".format(sub_data))
    transformed.append(inputter._transform_data(sub_data, mode))  # pylint: disable=protected-access

    for i, inputter in enumerate(self.inputters[1:]):
        with tf.variable_scope("inputter_{}".format(i+1)):
          sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i+1))
          transformed.append(inputter._transform_data(sub_data, mode))  # pylint: disable=protected-access

    if self.reducer is not None:
      transformed = self.reducer.reduce(transformed)
    return transformed

  def transform(self, inputs, mode):
    tf.logging.info(" >> [inputter.py class ParallelInputter transform]")
    transformed = super(ParallelInputter, self).transform(inputs, mode)
    if self.reducer is not None:
      transformed = self.reducer.reduce(transformed)
    return transformed

class HierarchicalInputter(ParallelInputter):
  """An multi inputter that process Hierarchical data."""

  def __init__(self, inputter_type, inputter_args, num, reducer=PackReducer(axis=1)):
    """Initializes a Hierarchical inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs. If
        set, Hierarchical inputs are assumed to have the same length.
    """
    vocabulary_file_key, embedding_size, embedding_file_key = inputter_args
    inputters = []
    self.num = num
    self.vocabulary_file_key = vocabulary_file_key
    self.embedding_file_key = embedding_file_key
    self.embedding_size = embedding_size

    for i in range(self.num):
      inputters.append(
        inputter_type(
          vocabulary_file_key=vocabulary_file_key,
          embedding_size=embedding_size,
          embedding_file_key=embedding_file_key))
    tf.logging.info(" >> [inputter.py class HierarchicalInputter __init__] inputters = {}".format("\n".join(["{}".format(x) for x in inputters])))
    tf.logging.info(" >> [inputter.py class HierarchicalInputter __init__] reducer = {}".format(reducer))
    super(HierarchicalInputter, self).__init__(inputters, reducer)

  def initialize(self, metadata):
    tf.logging.info(" >>>> [inputter.py Class HierarchicalInputter initialize]")
    self.vocabulary_file = metadata[self.vocabulary_file_key]
    self.embedding_file = metadata[self.embedding_file_key] if self.embedding_file_key else None
    self.num_oov_buckets = self.inputters[0].num_oov_buckets
    for idx, inputter in enumerate(self.inputters):
      tf.logging.info(" >>>> [inputter.py Class HierarchicalInputter initialize] initializing inputter ---- {} ---- ({}) ".format(idx, inputter))
      inputter.initialize(metadata)

  def process(self, data):
    """Prepares raw data.
    simply removed hooks here, inputters inside will call hooks themselves
    Args:
      data: The raw data.

    Returns:
      A dictionary of ``tf.Tensor``.

    See Also:
      :meth:`opennmt.inputters.inputter.Inputter.transform_data`
    """
    global debug_cnt
    debug_cnt += 1
    tf.logging.info(" >> [inputter.py class HierarchicalInputter process] data = self._process(data) debug_cnt = {}".format(debug_cnt))
    data = self._process(data)
    tf.logging.info(" >> [inputter.py class HierarchicalInputter process] after data = self._process(data)\ndata : \n{}".format("\n".join(["{}".format(x) for x in data.items()])))
    for key in self.volatile:
      data = self.remove_data_field(data, key)
    self.volatile.clear()
    tf.logging.info(" >> [inputter.py class HierarchicalInputter process] after remove_data_field()\ndata : \n{}".format("\n".join(["{}".format(x) for x in data.items()])))
    return data

  def _process(self, data):
    tf.logging.info(" >> [inputter.py class HierarchicalInputter _process] data = {}".format(data))
    processed_data = {}
    for i, inputter in enumerate(self.inputters):
      sub_data = inputter._process(data[i])  # pylint: disable=protected-access
      tf.logging.info(" >> [inputter.py class HierarchicalInputter _process No.{}] BEFORE sub_data : \n{}".format(i, "\n".join(["{}".format(x) for x in sub_data.items()])))
      tf.logging.info(" >> [inputter.py class HierarchicalInputter _process No.{}] inputter.volatile = {}".format(i, inputter.volatile))
      tf.logging.info(" >> [inputter.py class HierarchicalInputter _process No.{}] inputter.process_hooks = {}".format(i, inputter.process_hooks))
      for hook in inputter.process_hooks:
        sub_data = hook(inputter, sub_data)
      tf.logging.info(" >> [inputter.py class HierarchicalInputter _process No.{}] AFTER sub_data : \n{}".format(i, "\n".join(["{}".format(x) for x in sub_data.items()])))
      for key, value in six.iteritems(sub_data):
        prefixed_key = "inputter_{}_{}".format(i, key)
        processed_data = self.set_data_field(
            processed_data,
            prefixed_key,
            value,
            volatile=key in inputter.volatile)
      tf.logging.info(" >> [inputter.py class HierarchicalInputter _process No.{}] self.volatile = {}".format(i, self.volatile))
    return processed_data

  def _transform_data(self, data, mode):
    tf.logging.info(" >> [inputter.py class HierarchicalInputter _transform_data] data = {}".format(data))
    transformed = []

    '''
    All inputters in sub_inputter are of the same type, WordEmbedder, 
    and all reuse the same embedding as the inputter_1 in feature_inputter
    '''
    for i, inputter in enumerate(self.inputters):
      tf.logging.info(" >> [inputter.py class HierarchicalInputter _transform_data] inputter[{}] = {}".format(i, inputter))
      sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
      transformed.append(inputter._transform_data(sub_data, mode))  # pylint: disable=protected-access

    tf.logging.info(" >> [inputter.py class HierarchicalInputter _transform_data] transformed = {}".format(transformed))

    if self.reducer is not None:
      transformed = self.reducer.reduce(transformed)
    return transformed

  def _transform_sub_labels(self, data):
    transformed = []
    for i, inputter in enumerate(self.inputters):
        sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
        transformed.append(sub_data["ids_out"])
    tf.logging.info(" >> [inputter.py class HierarchicalInputter _transform_sub_labels] BEFORE transformed = {}".format(transformed))
    if self.reducer is not None:
      transformed = [tf.expand_dims(x, axis=-1) for x in transformed]
      transformed = self.reducer.reduce(transformed, has_depth=False)
      transformed = tf.squeeze(transformed, axis=-1)
    tf.logging.info(" >> [inputter.py class HierarchicalInputter _transform_sub_labels] AFTER transformed = {}".format(transformed))
    data = self.set_data_field(data, "ids_out", transformed, volatile=False)
    return data

  def transform(self, inputs, mode):
    tf.logging.info(" >> [inputter.py class HierarchicalInputter transform] inputs = {}".format(inputs))
    '''
      All inputter share the same embedding, use the 0th to lookup is fine
    '''
    return self.inputters[0].transform(inputs, mode) # word embedder

  def visualize(self, log_dir):
    tf.logging.info(" >>>> [inputter.py Class HierarchicalInputter visualize]")
    for inputter in self.inputters:
      inputter.visualize(log_dir)

  def get_length(self, data, to_reduce=False):
    tf.logging.info(" >> [inputter.py class HierarchicalInputter get_length]")
    lengths = []
    for i, inputter in enumerate(self.inputters):
      sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
      lengths.append(inputter.get_length(sub_data))
    if self.reducer is None:
      return lengths
    elif not to_reduce:
      return lengths[0]
    else:
      return self.reducer.pack_sequence_lengths(lengths)

  def get_vocab_size(self):
    return self.inputters[0].vocabulary_size

  def add_process_hooks(self, hooks):
    """Adds processing hooks.

    Processing hooks are additional and model specific data processing
    functions applied after calling this inputter
    :meth:`opennmt.inputters.inputter.Inputter.process` function.

    Args:
      hooks: A list of callables with the signature
        ``(inputter, data) -> data``.
    """
    tf.logging.info(" >> [inputter.py class HierarchicalInputter add_process_hooks]")
    for inputter in self.inputters:
      inputter.process_hooks.extend(hooks)

class MixedInputter(MultiInputter):
  """An multi inputter that applies several transformation on the same data."""

  def __init__(self,
               inputters,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs.
      dropout: The probability to drop units in the merged inputs.
    """
    '''
    Difference: 
    (0) Most fundamental: e,g, MixedInputter applies different text_inputter to the same dataset
    (1) MixedInputter assumes to use reducer and dropout
    (2) ParallelInputter set different variable scope for [inputters]
    '''
    super(MixedInputter, self).__init__(inputters)
    self.reducer = reducer
    self.dropout = dropout

  def get_length(self, data):
    return self.inputters[0].get_length(data)

  def make_dataset(self, data_file):
    tf.logging.info(" >> [inputter.py class MixedInputter make_dataset]")
    return self.inputters[0].make_dataset(data_file)

  def get_dataset_size(self, data_file):
    return self.inputters[0].get_dataset_size(data_file)

  def _get_serving_input(self):
    all_receiver_tensors = {}
    all_features = {}
    for inputter in self.inputters:
      receiver_tensors, features = inputter._get_serving_input()  # pylint: disable=protected-access
      all_receiver_tensors.update(receiver_tensors)
      all_features.update(features)
    return all_receiver_tensors, all_features

  def _process(self, data):
    tf.logging.info(" >> [inputter.py class MixedInputter _process]")
    for inputter in self.inputters:
      data = inputter._process(data)  # pylint: disable=protected-access
      self.volatile |= inputter.volatile
    return data

  def _transform_data(self, data, mode):
    tf.logging.info(" >> [inputter.py class MixedInputter _transform_data]")
    transformed = []
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        transformed.append(inputter._transform_data(data, mode))  # pylint: disable=protected-access
    outputs = self.reducer.reduce(transformed)
    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs

  def transform(self, inputs, mode):
    tf.logging.info(" >> [inputter.py class MixedInputter transform]")
    transformed = super(MixedInputter, self).transform(inputs, mode)
    outputs = self.reducer.reduce(transformed)
    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs
