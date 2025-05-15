"""
some operations for model
author: ErnestinaQiu
"""
import paddle
import paddle.nn.functional as F
from typing import List
from exp.model.ties.caloGraphNN import indexing_tensor


class DenseLayer(paddle.nn.Layer):
    def __init__(self, input_dim, output_dim):
        super(DenseLayer, self).__init__()
        self.linear = paddle.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


def gather_features_from_conv_head(conv_head, vertices_y, vertices_x, vertices_y2, vertices_x2, scale_y, scale_x):
    """Gather features from a 2D image.

    Args:
        conv_head (_type_): The 2D conv head with shape [batch, height, width, channels]
        vertices_y (int): The y position of each of the vertex with shape [batch, max_vertices]
        vertices_x (int): The x position of each of the vertex with shape [batch, max_vertices]
        vertices_y2 (int): The height of each of the vertex with shape [batch, max_vertices]
        vertices_x2 (int): The width of each of the feature with shape [batch, max_vertices]
        scale_y (float): A scalar to show y_scale
        scale_x (float): A scalar to show x_scale
    Returns:
        features (): the gathered features with shape [batch, max_vertices, channels]
    """
    # normalization


def edge_conv_layer(vertices_in: paddle.Tensor, num_neighbors: int = 30, mpl_layers: List = [64, 64, 64], aggregation_method = paddle.max, share_keyword=None, edge_activation=None):
    """_summary_

    Args:
        vertices_in (paddle.Tensor)
        num_neighbors (int, optional): _description_. Defaults to 30.
        mpl_layers (list, optional): _description_. Defaults to [64, 64, 64].
        aggregation_method (str, optional): _description_. Defaults to 'dim2_max'.
        share_keyword (_type_, optional): _description_. Defaults to None.
        edge_activation (_type_, optional): _description_. Defaults to None.
    """
    trans_space = vertices_in
    indexing, _ = indexing_tensor(trans_space, num_neighbors)
    # change indexing to be not self-referential
    neighbour_space = paddle.gather_nd(vertices_in, indexing)

    expanded_trans_space = paddle.unsqueeze(trans_space, axis=2)
    expanded_trans_space = paddle.tile(expanded_trans_space, [1, 1, num_neighbors, 1])

    diff = expanded_trans_space - neighbour_space
    edge = paddle.concat([expanded_trans_space, diff], axis=-1)

    for out_dim in mpl_layers:
        dense_layer = DenseLayer(input_dim=edge.shape[1], output_dim=out_dim)
        edge = dense_layer(edge)

    if edge_activation is not None:
        edge = edge_activation(edge)

    vertex_out = aggregation_method(edge, axis=2)

    return vertex_out

