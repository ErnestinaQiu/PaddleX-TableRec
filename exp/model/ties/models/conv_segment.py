""" 
Convolution Network
author: ErnestinaQiu
"""
import paddle
import numpy as np


class BasicConvSegment(paddle.nn.Layer):
    def __init__(self, normalized_width: int, normalized_height: int, name_scope=None, dtype="float32"):
        super().__init__(name_scope, dtype)
        self.normalized_width = normalized_width
        self.normalized_height = normalized_height
        self.con2d_2 = paddle.nn.Conv2D(
            in_channels=10, out_channels=10, kernel_size=(3, 3), stride=1, padding=0, bias_attr=True
        )

    def forward(self, x):
        """_summary_

        Args:
            x (tensor): Images with shape (batch, channel, height, width)

        Returns:
            tensor: image features with shape (batch, 10, height - 12, width - 12)
        """
        assert len(x.shape) == 4, "Input must be 4D."
        assert x.shape[2] == self.normalized_height and x.shape[3] == self.normalized_width, f"Image shape must be [{self.normalized_height}, {self.normalized_width}], but got [{x.shape[2]}, {x.shape[3]}]"

        if isinstance(x, np.ndarray):
            x = paddle.to_tensor(x)

        self.con2d_1 = paddle.nn.Conv2D(
            in_channels=x.shape[1], out_channels=10, kernel_size=(3, 3), stride=1, padding=0, bias_attr=True
        )

        _graph_from_image = self.con2d_1(x)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        return _graph_from_image
