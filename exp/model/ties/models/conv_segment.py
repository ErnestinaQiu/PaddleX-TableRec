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
        self.con2d_1 = paddle.nn.Conv2D(
            in_channels=1, out_channels=10, kernel_size=(3, 3), stride=1, padding=0, bias_attr=True
        )
        self.con2d_2 = paddle.nn.Conv2D(
            in_channels=10, out_channels=10, kernel_size=(3, 3), stride=1, padding=0, bias_attr=True
        )

    def forward(self, x):
        assert len(x.shape) == 4, "Input must be 4D."

        if isinstance(x, np.ndarray):
            x = paddle.to_tensor(x)

        new_xs = []
        for _x in x:
            _x = _x.reshape((x.shape[1], x.shape[2], x.shape[3]))
            new_x = paddle.vision.transforms.resize(_x, size=(256, 256))
            new_xs.append(new_x)
        x = np.array(new_xs)
        x = paddle.to_tensor(x)
        x = x.astype(dtype='float32')

        _graph_from_image = self.con2d_1(x)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        _graph_from_image = self.con2d_2(_graph_from_image)
        return _graph_from_image
