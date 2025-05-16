""" 
Dgcnn Network
author: ErnestinaQiu
"""
import paddle
from exp.model.ties.caloGraphNN import high_dim_dense, layer_global_exchange
from exp.model.ties.ops import edge_conv_layer, DenseLayer


class DgcnnSegment(paddle.nn.Layer):
    def __init__(self, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)

    def forward(self, x):
        self.bn = paddle.nn.BatchNorm2D(num_features=x.shape(1), momentum=0.8)
        feat = self.bn(x)
        # global transform to 3D
        feat = high_dim_dense(feat, nodes=64)

        feat = edge_conv_layer(feat, 10, [64, 64, 64])
        feat_g = layer_global_exchange(feat)
        feat = paddle.concat([feat, feat_g], axis=-1)
        dense_net = DenseLayer(input_dim=feat.shape[1], output_dim=64)
        feat = dense_net(feat)

        feat1 = edge_conv_layer(feat, 10, [64, 64, 64])
        feat1_g = layer_global_exchange(feat1)
        feat1 = paddle.concat([feat1, feat1_g], axis=-1)
        dense_net2 = DenseLayer(input_dim=feat1.shape[1], output_dim=64)
        feat1 = dense_net2(feat1)

        feat2 = edge_conv_layer(feat1, 10, [64, 64, 64])
        feat2_g = layer_global_exchange(feat2)
        feat2 = paddle.concat([feat2, feat2_g], axis=-1)
        dense_net3 = DenseLayer(input_dim=feat2.shape[1], output_dim=64)
        feat2 = dense_net3(feat2)

        feat3 = edge_conv_layer(feat2, 10, [64, 64, 64])

        feat = paddle.concat([feat, feat1, feat2, feat_g, feat1_g, feat2_g, feat3], axis=-1)
        dense1 = DenseLayer(input_dim=feat.shape[1], output_dim=128)
        feat = dense1(feat)
        dense2 = DenseLayer(input_dim=feat.shape[1], output_dim=128)
        feat = dense2(feat)
        return feat
