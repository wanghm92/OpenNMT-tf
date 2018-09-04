"""Module defining decoders."""

from opennmt.decoders.rnn_decoder import RNNDecoder
from opennmt.decoders.rnn_decoder import AttentionalRNNDecoder
from opennmt.decoders.rnn_decoder import MultiAttentionalRNNDecoder
from opennmt.decoders.hierarchical_rnn_decoder import HierarchicalAttentionalRNNDecoder
from opennmt.decoders.tf_contrib_seq2seq_decoder import TfContribSeq2seqDecoder
from opennmt.decoders.basic_decoder import BasicDecoder

from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
