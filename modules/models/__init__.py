"""
Import all the modules in the package
"""

from ._base import BaseModelWrapper
from .mls import CommonMachineLearning
from .voting import Voting
from .jeong import JeongStacking, JeongStage
# from .arima import ARIMA
from .cnn1d import ConvolutionNetwork1D
from .cnn_attention import CNNAttention
from .attention import AttentionModel
from .multi_attention import MultiAttentionModel
# from .gans import GANs
from .transformer import TransformerTS
# from .kan import KAN
from .batri.batri_hum_cnn1d import BaTriHumCNN1D
from .cnn1d_v3 import BaTriTempCNN1D
from .cnn1d_v2 import PhuLienCNN1D
