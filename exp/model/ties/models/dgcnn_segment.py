""" 
Dgcnn Network
author: ErnestinaQiu
"""
import paddle


class Dgcnn(paddle.nn.Layer):
    def __init__(self, name_scope = None, dtype = "float32"):
        super().__init__(name_scope, dtype)


    def forward(self, x):
        self.bn = paddle.nn.BatchNorm2D(num_features=x.shape(1), momentum=0.8)


        feat = 
        return 