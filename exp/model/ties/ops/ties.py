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
        conv_head (paddle.Tensor): the output from convolution network
        vertices_y (paddle.Tensor): The y position of each of the vertex with shape [batch, max_vertices]
        vertices_x (paddle.Tensor: The x position of each of the vertex with shape [batch, max_vertices]
        vertices_y2 (paddle.Tensor): The height of each of the vertex with shape [batch, max_vertices]
        vertices_x2 (paddle.Tensor): The width of each of the feature with shape [batch, max_vertices]
        scale_y (paddle.Tensor): A scalar to show y_scale
        scale_x (paddle.Tensor): A scalar to show x_scale

    Returns:
        The gathered features with shape [batch, max_vertices, channels]
    """
    vertices_y = paddle.cast(vertices_y, paddle.float32) * scale_y
    vertices_x = paddle.cast(vertices_x, paddle.float32) * scale_x
    vertices_y2 = paddle.cast(vertices_y2, paddle.float32) * scale_y
    vertices_x2 = paddle.cast(vertices_x2, paddle.float32) * scale_x

    batch_size, max_vertices = vertices_y.shape
    batch_size, max_vertices = int(batch_size.value), int(max_vertices.value)

    batch_range = paddle.arange(0, batch_size, dtype=paddle.float32).unsqueeze(-1).unsqueeze(-1)
    # transform the dimension to fit max_vertices
    batch_range = paddle.tile(batch_range, repeat_times=[1, max_vertices, 1])

    mid_y = (vertices_y + vertices_y2) / 2.0
    mid_x = (vertices_x + vertices_x2) / 2.0

    mid_y = mid_y.unsqueeze(-1)
    mid_x = mid_x.unsqueeze(-1)

    indexing_tensor = paddle.concat([batch_range, mid_y, mid_x], axis=-1)
    indexing_tensor = paddle.cast(indexing_tensor, paddle.int64)

    return paddle.gather_nd(conv_head, indexing_tensor)


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
