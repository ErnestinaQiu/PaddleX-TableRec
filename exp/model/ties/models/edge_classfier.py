"""
author: ErnestinaQiu
"""
import paddle
from paddle import nn


class EdgeClassifier(nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        