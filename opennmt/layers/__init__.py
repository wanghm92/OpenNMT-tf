"""Module defining reusable and model specific layers."""

from opennmt.layers.reducer import SumReducer
from opennmt.layers.reducer import MultiplyReducer
from opennmt.layers.reducer import ConcatReducer
from opennmt.layers.reducer import JoinReducer

from opennmt.layers.bridge import CopyBridge
from opennmt.layers.bridge import ZeroBridge
from opennmt.layers.bridge import DenseBridge
from opennmt.layers.bridge import AttentionWrapperStateAggregatedDenseBridge
from opennmt.layers.bridge import AttentionWrapperStatePairwiseDenseBridge
from opennmt.layers.bridge import AttentionWrapperStatePairwiseWeightedSumBridge
from opennmt.layers.bridge import AttentionWrapperStatePairwiseGatingBridge
from opennmt.layers.bridge import AttentionWrapperStateAggregatedGatingBridge
from opennmt.layers.bridge import AttentionWrapperStateAverageBridge


from opennmt.layers.position import PositionEmbedder
from opennmt.layers.position import SinusoidalPositionEncoder
