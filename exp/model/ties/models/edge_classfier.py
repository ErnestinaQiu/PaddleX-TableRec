"""
author: ErnestinaQiu
"""
import paddle
from paddle import nn
import paddle.nn.functional as F
from exp.model.ties.ops import edge_conv_layer, DenseLayer


class EdgeClassifier(nn.Layer):
    def __init__(self, dim_num_vertices, max_vertices, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.dim_num_vertices = dim_num_vertices
        self.max_vertices = max_vertices

    def forward(self, graph_features, training=True, *inputs, **kwargs):
        """_summary_

        Args:
            graph_features (paddle.Tensor): From dgcnn segment network, with shape (batch, )
        Returns:

        """
        if len(graph_features.shape) == 4:
            self.batch_norm = nn.BatchNorm2D(num_features=graph_features.shape[1], momentum=0.8)
        elif len(graph_features.shape) == 3:
            self.batch_norm = nn.BatchNorm1D(num_features=graph_features.shape[1], momentum=0.8)
        elif len(graph_features.shape) == 5:
            self.batch_norm = nn.BatchNorm3D(num_features=graph_features.shape[1], momentum=0.8)

        net = self.batch_norm(graph_features)
        self.dense1 = DenseLayer(output_dim=256)
        net = self.dense1(net)
        net = self.dense1(net)
        net = self.dense1(net)

        self.dense2 = DenseLayer(output_dim=900)
        net = self.dense2(net)

        predicted_adj_matrix = F.sigmoid(net)

        return predicted_adj_matrix
