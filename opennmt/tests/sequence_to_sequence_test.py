import tensorflow as tf

from opennmt.models import sequence_to_sequence


class SequenceToSequenceTest(tf.test.TestCase):

  def testReplaceUnknownTarget(self):
    target_tokens = [
      ["Hello", "world", "!", "", "", ""],
      ["<unk>", "name", "is", "<unk>", ".", ""]]
    source_tokens = [
      ["Bonjour", "le", "monde", "!", ""],
      ["Mon", "nom", "est", "Max", "."]]
    attention = [
      [[0.9, 0.1, 0.0, 0.0, 0.0],
       [0.2, 0.1, 0.7, 0.0, 0.0],
       [0.0, 0.1, 0.1, 0.8, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0]],
      [[0.8, 0.1, 0.1, 0.0, 0.0],
       [0.1, 0.9, 0.0, 0.0, 0.0],
       [0.0, 0.1, 0.8, 0.1, 0.0],
       [0.1, 0.1, 0.2, 0.6, 0.0],
       [0.0, 0.1, 0.1, 0.3, 0.5],
       [0.0, 0.0, 0.0, 0.0, 0.0]]]
    replaced_target_tokens = sequence_to_sequence.replace_unknown_target(
        target_tokens,
        source_tokens,
        attention,
        unknown_token="<unk>")
    with self.test_session() as sess:
      replaced_target_tokens = sess.run(replaced_target_tokens)
      self.assertNotIn(b"<unk>", replaced_target_tokens.flatten().tolist())
      self.assertListEqual(
          [b"Hello", b"world", b"!", b"", b"", b""], replaced_target_tokens[0].tolist())
      self.assertListEqual(
          [b"Mon", b"name", b"is", b"Max", b".", b""], replaced_target_tokens[1].tolist())

  def _testPharaohAlignments(self, line, lengths, expected_matrix):
    matrix = sequence_to_sequence.alignment_matrix_from_pharaoh(
        tf.constant(line), lengths[0], lengths[1])
    matrix = tf.cast(matrix, tf.int32)
    with self.test_session() as sess:
      self.assertListEqual(expected_matrix, sess.run(matrix).tolist())

   def testPharaohAlignments(self):
    self._testPharaohAlignments("0-0", [1, 1], [[1]])
    self._testPharaohAlignments(
        "0-0 1-1 2-2 3-3", [4, 4], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    self._testPharaohAlignments(
        "0-0 1-1 2-3 3-2", [4, 4], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    self._testPharaohAlignments(
        "0-0 1-2 1-1", [2, 3], [[1, 0], [0, 1], [0, 1]])
    self._testPharaohAlignments(
        "0-0 1-2 1-1 2-4", [3, 5], [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])

if __name__ == "__main__":
  tf.test.main()
