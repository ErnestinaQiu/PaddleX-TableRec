"""
author: ErnestinaQiu
"""
import paddle
from paddle import nn
import paddle.nn.functional as F
from exp.model.ties.ops import edge_conv_layer, DenseLayer


class EdgeClassifier(nn.Layer):
    def __init__(self, config, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.dim_num_vertices = config['dim_num_vertices']
        self.max_vertices = config['max_vertices']

    def forward(self, graph_features, gt_matrix, global_features, training=False, *inputs, **kwargs):
        """_summary_

        Args:
            graph_features (paddle.Tensor): From dgcnn segment network
            gt_matrix (paddle.Tensor): with shape (num_batch, max_vertices, max_vertices)
            global_features (paddle.Tensor): with shape (num_batch, num_global_features)
        Returns:

        """
        if len(graph_features.shape) == 4:
            self.batch_norm = nn.BatchNorm2D(num_features=graph_features.shape[1], momentum=0.8)
        elif len(graph_features.shape) == 3:
            self.batch_norm = nn.BatchNorm1D(num_features=graph_features.shape[1], momentum=0.8)
        elif len(graph_features.shape) == 5:
            self.batch_norm = nn.BatchNorm3D(num_features=graph_features.shape[1], momentum=0.8)

        net = self.batch_norm(graph_features, training=training)
        self.dense1 = DenseLayer(input_dim=net.shape[1], output_dim=256)
        net = self.dense1(net)
        net = self.dense1(net)
        net = self.dense1(net)

        self.dense2 = DenseLayer(input_dim=net.shape[1], output_dim=2)
        net = self.dense2(net)

        num_features = global_features[:, self.dim_num_vertices]

        # don't understand the use of the flowing mask
        mask = F.sequence_mask(num_features, maxlen=self.max_vertices)
        mask = paddle.unsqueeze(mask, axis=-1)
        mask = paddle.cast(x=mask, dtype=paddle.float32)

        predicted_adj_matrix = paddle.argmax(net, axis=-1)

        return predicted_adj_matrix
