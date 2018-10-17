"""Module defining reusable and model specific layers."""

from opennmt.layers.reducer import SumReducer
from opennmt.layers.reducer import MultiplyReducer
from opennmt.layers.reducer import ConcatReducer
from opennmt.layers.reducer import JoinReducer

from opennmt.layers.bridge import CopyBridge
from opennmt.layers.bridge import ZeroBridge
from opennmt.layers.bridge import DenseBridge
from opennmt.layers.nested_state_bridge import NestedStateAggregatedDenseBridge
from opennmt.layers.nested_state_bridge import NestedStatePairwiseDenseBridge
from opennmt.layers.nested_state_bridge import NestedStatePairwiseWeightedSumBridge
from opennmt.layers.nested_state_bridge import NestedStatePairwiseGatingBridge
from opennmt.layers.nested_state_bridge import NestedStateAggregatedGatingBridge
from opennmt.layers.nested_state_bridge import NestedStateAverageBridge


from opennmt.layers.position import PositionEmbedder
from opennmt.layers.position import SinusoidalPositionEncoder
