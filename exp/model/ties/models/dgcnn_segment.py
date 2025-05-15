""" 
Dgcnn Network
author: ErnestinaQiu
"""
import paddle
from exp.model.ties.caloGraphNN import high_dim_dense
from exp.model.ties.ops import edge_conv_layer


class Dgcnn(paddle.nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)

    def forward(self, x):
        self.bn = paddle.nn.BatchNorm2D(num_features=x.shape(1), momentum=0.8)
        x = self.bn(x)
        # global transform to 3D
        x = high_dim_dense(x, nodes=64)

        x = edge_conv_layer(x, 10, [64, 64, 64])

        return 