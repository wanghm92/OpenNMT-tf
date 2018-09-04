import collections, sys, os
sys.path.append(os.getcwd())

import tensorflow as tf

from opennmt.layers import reducer


class ReducerTest(tf.test.TestCase):


  def testPadWithIdentity(self):
    tensor = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    expected = [
        [[1], [1], [1], [1], [0]],
        [[1], [2], [3], [1], [1]],
        [[1], [2], [0], [0], [0]]]
    lengths = [1, 3, 2]
    max_lengths = [4, 5, 2]

    padded, mask, mask_combined, identity_mask = reducer.pad_with_identity(
        tf.constant(tensor, dtype=tf.float32),
        tf.constant(lengths),
        tf.constant(max_lengths),
        identity_values=1)

    self.assertEqual(1, padded.get_shape().as_list()[-1])

    with self.test_session() as sess:
        padded, mask, mask_combined, identity_mask = sess.run([padded, mask, mask_combined, identity_mask])
        print(padded)
        print(mask)
        print(mask_combined)
        print(identity_mask)
        self.assertAllEqual(expected, padded)

  def testPadWithIdentityWithMaxTime(self):
    tensor = [
        [[1], [-1], [-1], [-1]],
        [[1], [2], [3], [-1]],
        [[1], [2], [-1], [-1]]]
    expected = [
        [[1], [1], [1], [1], [0], [0]],
        [[1], [2], [3], [1], [1], [0]],
        [[1], [2], [0], [0], [0], [0]]]
    lengths = [1, 3, 2]
    max_lengths = [4, 5, 2]
    maxlen = 6

    padded, mask, mask_combined, identity_mask = reducer.pad_with_identity(
        tf.constant(tensor, dtype=tf.float32),
        tf.constant(lengths),
        tf.constant(max_lengths),
        identity_values=1,
        maxlen=maxlen)

    self.assertEqual(1, padded.get_shape().as_list()[-1])

    with self.test_session() as sess:
      padded, mask, mask_combined, identity_mask = sess.run([padded, mask, mask_combined, identity_mask])
      print(padded)
      print(mask)
      print(mask_combined)
      print(identity_mask)
      self.assertAllEqual(expected, padded)

 
if __name__ == "__main__":
  tf.test.main()
