import os
import cv2
import random
from typing import Dict

import paddle
from paddle import nn

from exp.model.ties.models.conv_segment import BasicConvSegment
from exp.model.ties.models.dgcnn_segment import DgcnnSegment
from exp.model.ties.models.edge_classfier import EdgeClassifier
from exp.model.ties.ops.ties import gather_features_from_conv_head


class BasicModel(nn.Layer):
    def __init__(self, config: Dict):
        # the following must be from config in the future version
        self.max_vertices = config['max_vertices']

        self.normalized_width = config['normalized_width']
        self.normalized_height = config['normalized_height']

        self.num_vertex_features = config['num_vertex_features']
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.max_words_len = config['max_words_len']
        self.num_batch = config['num_batch']
        self.num_global_features = config['num_global_features']
        self.image_channels = config['image_channels']
        self.dim_vertex_x_position = config['dim_vertex_x_position']
        self.dim_vertex_y_position = config['dim_vertex_y_position']
        self.dim_vertex_x2_position = config['dim_vertex_x2_position']
        self.dim_vertex_y2_position = config['dim_vertex_y2_position']

        self.dim_num_vertices = config['dim_num_vertices']
        self.samples_per_vertex = config['samples_per_vertex']
        self.variable_scope = config['variable_scope']
        self.learning_rate = config['learning_rate']

        self.is_sampling_balanced = config['is_sampling_balanced']

        self.momentum = config['momentum']

        self.conv_segment = BasicConvSegment(normalized_height=self.normalized_height, normalized_width=self.normalized_width)
        self.graph_segment = DgcnnSegment()
        self.cell_clas_model = EdgeClassifier(dim_num_vertices=self.dim_num_vertices, max_vertices=self.max_vertices)
        self.row_clas_model = EdgeClassifier(dim_num_vertices=self.dim_num_vertices, max_vertices=self.max_vertices)
        self.col_clas_model = EdgeClassifier(dim_num_vertices=self.dim_num_vertices, max_vertices=self.max_vertices)

    def forward(self, x: Dict, **kwargs):
        """

        Args:
            x (dict): {"images": paddle.Tensor|[b, c, h, w], "text_boxes": list|[[text boxes in one image], [...]]},
                    images with shape as [batch, channel, width, height],
                    text_box with shape [x1, y1, x2, y2]

        Returns:
            probability (paddle.Tensor): the probability of the 
        """
        images = x['images']
        b, c, h, w = images.shape

        assert len(images.shape) == 4, "Input images must be 4D tensor."
        assert len(x['text_boxes']) == len(x['text_words_length']) == b, f"len(x['text_boxes']) != len(x['text_words_length']), len(x['text_boxes']): {len(x['text_boxes'])}, len(x['text_words_length']): {len(x['text_words_length'])}"

        text_boxes = x['text_boxes']
        vertices_y = []
        vertices_y2 = []
        vertices_x = []
        vertices_x2 = []
        for box in text_boxes:
            vertices_x.append(box[0])
            vertices_y.append(box[1])
            vertices_x2.append(box[2])
            vertices_y2.append(box[3])

        vertices_x = paddle.to_tensor(vertices_x, dtype=paddle.float32)
        vertices_x = paddle.reshape(vertices_x, (vertices_x.shape[0], 1))

        vertices_y = paddle.to_tensor(vertices_y, dtype=paddle.float32)
        vertices_y = paddle.reshape(vertices_y, (vertices_y.shape[0], 1))

        vertices_x2 = paddle.to_tensor(vertices_x2, dtype=paddle.float32)
        vertices_x2 = paddle.reshape(vertices_x2, (vertices_x2.shape[0], 1))

        vertices_y2 = paddle.to_tensor(vertices_y2, dtype=paddle.float32)
        vertices_y2 = paddle.reshape(vertices_y2, (vertices_y2.shape[0], 1))

        conv_head = self.conv_segment(x)

        _, post_height, post_width, _ = conv_head.shape
        scale_y = float(post_height) / float(h)
        scale_x = float(post_width) / float(h)

        gathered_image_features = gather_features_from_conv_head(conv_head, vertices_y, vertices_x,
                                                                 vertices_y2, vertices_x2, scale_y, scale_x)

        _graph_vertex_features = paddle.zeros(shape=(b, self.max_vertices, self.num_vertex_features), dtype=paddle.float32)
        words_length = x['text_words_length']
        for i in range(b):
            assert len(text_boxes[i]) == len(words_length[i]), f'len(text_boxes[{i}]) != len(words_length[{i}]) in batch {i}, len(text_boxes[{i}]): {len(text_boxes[i])}, len(words_length[{i}]): {len(words_length[i])}'
            for j in range(len(text_boxes[i])):
                x1, y1, x2, y2 = text_boxes[i][j]
                _graph_vertex_features[i, j, :4] = [x1, y1, x2, y2]

        vertices_combined_features = paddle.concat((_graph_vertex_features, gathered_image_features), axis=-1)

        graph_features = self.graph_segment(vertices_combined_features)

        cell_predicted_adj_matrix = self.cell_clas_model(graph_features)
        row_predicted_adj_matrix = self.row_clas_model(graph_features)
        col_predicted_adj_matrix = self.col_clas_model(graph_features)

        return {'cell_predicted_adj_matrix': cell_predicted_adj_matrix, 'row_predicted_adj_matrix': row_predicted_adj_matrix, 'col_predicted_adj_matrix': col_predicted_adj_matrix}

    def set_conv_segment(self, conv_segment):
        self.conv_segment = conv_segment

    def set_graph_segment(self, dgcnn_segment):
        self.graph_segment = dgcnn_segment


