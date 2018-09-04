"""Custom hooks."""

from __future__ import print_function

import io
import time
import six
import pprint
import tensorflow as tf

from opennmt.utils import misc
pp = pprint.PrettyPrinter(indent=4)

class LogParametersCountHook(tf.train.SessionRunHook):
  """Simple hook that logs the number of trainable parameters."""

  def begin(self):
    tf.logging.info("Number of trainable parameters: %d", misc.count_parameters())


_DEFAULT_COUNTERS_COLLECTION = "counters"


def add_counter(name, tensor):
  """Registers a new counter.

  Args:
    name: The name of this counter.
    tensor: The integer ``tf.Tensor`` to count.

  See Also:
    :meth:`opennmt.utils.misc.WordCounterHook` that fetches these counters
    to log their value in TensorBoard.
  """
  count = tf.cast(tensor, tf.int64)
  total_count_init = tf.Variable(
      initial_value=0,
      name=name + "_init",
      trainable=False,
      dtype=count.dtype)
  total_count = tf.assign_add(
      total_count_init,
      count,
      name=name)
  tf.add_to_collection(_DEFAULT_COUNTERS_COLLECTION, total_count)


class CountersHook(tf.train.SessionRunHook):
  """Hook that summarizes counters.

  Implementation is mostly copied from StepCounterHook.
  """
  '''
  The run_values argument contains results of requested ops/tensors by before_run().
  The run_context argument is 
  '''
  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir

  def begin(self):
    self._counters = tf.get_collection(_DEFAULT_COUNTERS_COLLECTION)
    if not self._counters:
      return

    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)

    self._last_count = [None for _ in self._counters]
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use WordCounterHook.")

    self._debug_ops = misc.get_dict_from_collection("debug")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    '''
    Called before each call to run().
    You can return from this call a SessionRunArgs object indicating ops or tensors to add to the upcoming run() call.
    These ops/tensors will be run together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run() call.
    SessionRunArgs Represents arguments to be added to a Session.run() call
      fetches: Exactly like the 'fetches' argument to Session.Run().
        Can be a single tensor or op, a list of 'fetches' or a dictionary of fetches.
          For example: fetches = global_step_tensor
                       fetches = [train_op, summary_op, global_step_tensor]
                       fetches = {'step': global_step_tensor, 'summ': summary_op}
                       fetches = {'step': global_step_tensor, 'ops': [train_op, check_nan_op]}
      feed_dict: Exactly like the feed_dict argument to Session.Run()
      options: Exactly like the options argument to Session.run(), i.e., a config_pb2.RunOptions proto.

    The run_context argument is a SessionRunContext that provides information about the upcoming run() call:
        the originally requested op/tensors, the TensorFlow Session.

    At this point graph is finalized and you can not add ops.
    '''
    if not self._counters:
      return None
    return tf.train.SessionRunArgs([self._counters, self._global_step, self._debug_ops])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    '''
    Called after each call to run().
    If session.run() raises any exceptions then after_run() is not called.
    :param run_context: the same one send to before_run call.
                        run_context.request_stop() can be called to stop the iteration.
    :param run_values: results of requested ops/tensors by before_run()
    '''

    if not self._counters:return
    counters, step, debug_ops = run_values.results
    # tf.logging.info(" >> [hooks.py class CountersHook after_run] debug_ops :")
    # pp.pprint(debug_ops)
    if self._timer.should_trigger_for_step(step):
      elapsed_time, _ = self._timer.update_last_triggered_step(step)
      if elapsed_time is not None:
        for i in range(len(self._counters)):
          if self._last_count[i] is not None:
            name = self._counters[i].name.split(":")[0]
            value = (counters[i] - self._last_count[i]) / elapsed_time
            if self._summary_writer is not None:
              summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
              self._summary_writer.add_summary(summary, step)
            tf.logging.info("%s: %g", name, value)
          self._last_count[i] = counters[i]


class LogPredictionTimeHook(tf.train.SessionRunHook):
  """Hooks that gathers and logs prediction times."""

  def begin(self):
    self._total_time = 0
    self._total_tokens = 0
    self._total_examples = 0

  def before_run(self, run_context):
    self._run_start_time = time.time()
    predictions = run_context.original_args.fetches
    return tf.train.SessionRunArgs(predictions)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    self._total_time += time.time() - self._run_start_time
    predictions = run_values.results
    batch_size = next(six.itervalues(predictions)).shape[0]
    self._total_examples += batch_size
    length = predictions.get("length")
    if length is not None:
      if len(length.shape) == 2:
        length = length[:, 0]
      self._total_tokens += sum(length)

  def end(self, session):
    tf.logging.info("Total prediction time (s): %f", self._total_time)
    tf.logging.info("Average prediction time (s): %f", self._total_time / self._total_examples)
    if self._total_tokens > 0:
      tf.logging.info("Tokens per second: %f", self._total_tokens / self._total_time)


class SaveEvaluationPredictionHook(tf.train.SessionRunHook):
  """Hook that saves the evaluation predictions."""

  def __init__(self, model, output_file, post_evaluation_fn=None):
    """Initializes this hook.

    Args:
      model: The model for which to save the evaluation predictions.
      output_file: The output filename which will be suffixed by the current
        training step.
      post_evaluation_fn: (optional) A callable that takes as argument the
        current step and the file with the saved predictions.
    """
    self._model = model
    self._output_file = output_file
    self._post_evaluation_fn = post_evaluation_fn

  def begin(self):
    '''
    Called once before using the session.
      When called, the default graph is the one that will be launched in the session.
      The hook can modify the graph by adding new operations to it.
      After the begin() call the graph will be finalized
      and the other callbacks can not modify the graph anymore.
      Second call of begin() on the same graph, should not change the graph.
    '''
    self._predictions = misc.get_dict_from_collection("predictions")
    if not self._predictions:
      raise RuntimeError("The model did not define any predictions.")
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use SaveEvaluationPredictionHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs([self._predictions, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    predictions, self._current_step = run_values.results
    self._output_path = "{}.{}".format(self._output_file, self._current_step)
    with io.open(self._output_path, encoding="utf-8", mode="a") as output_file:
      # TODO: check this hook !!!
      for prediction in misc.extract_batches(predictions):
        self._model.print_prediction(prediction, stream=output_file)

  def end(self, session):
    tf.logging.info("Running _post_evaluation_fn (BLEU); Evaluation predictions saved to %s", self._output_path)
    if self._post_evaluation_fn is not None:
      self._post_evaluation_fn(self._current_step, self._output_path)
