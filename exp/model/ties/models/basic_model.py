import os
import cv2
import random
from typing import Dict

import paddle
from paddle import nn

from exp.model.ties.models.conv_segment import BasicConvSegment
from exp.model.ties.models.dgcnn_segment import DgcnnSegment
from exp.model.ties.ops.ties import gather_features_from_conv_head


class BasicModel(nn.Layer):
    def __init__(self, config: Dict):
        # the following must be from config in the future version
        self.normalized_width = 256
        self.normalized_height = 256

        self.num_vertex_features = 5
        self.image_height = 768
        self.image_width = 1366
        self.max_words_len = 30
        self.num_batch = 30
        self.num_global_features = 3
        self.image_channels = 1
        self.dim_vertex_x_position = 0
        self.dim_vertex_y_position = 1
        self.dim_vertex_x2_position = 2
        self.dim_vertex_y2_position = 3

        self.dim_num_vertices = 2
        self.samples_per_vertex = 6
        self.variable_scope = 'basic_conv_graph_alpha_1'
        self.learning_rate = 0.0001

        self.is_sampling_balanced = 1

        self.visual_feedback_out_path = 150

        self.momentum = 0.6

        self.conv_segment = BasicConvSegment(normalized_height=self.normalized_height, normalized_width=self.normalized_width)
        self.graph_segment = DgcnnSegment()

    def forward(self, x: Dict, **kwargs):
        """

        Args:
            x (dict): {"images": paddle.Tensor|numpy.ndarray, "text_boxes": list},
                    images with shape as [batch, channel, width, height],
                    text_box as [x1, y1, x2, y2]

        Returns:
            probability (paddle.Tensor): the probability of the 
        """
        assert len(x.shape) == 4, "Input must be 4D tensor."
        images = x['images']
        b, c, w, h = images.shape

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
        

        return


    def set_conv_segment(self, conv_segment):
        self.conv_segment = conv_segment

    def set_graph_segment(self, dgcnn_segment):
        self.graph_segment = dgcnn_segment


