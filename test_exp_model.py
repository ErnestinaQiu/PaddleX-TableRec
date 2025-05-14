"""
author: ErnestinaQiu
"""
import numpy as np
from exp.model.ties.models.conv_segment import BasicConvSegment
from exp.model.ties.models.dgcnn_segment import Dgcnn


class TestModel:
    def test_conv_segment(self):
        conv = BasicConvSegment()
        x = np.ones(shape=(1, 1, 700, 100))
        y = conv(x)
        print(y)
        print(y.shape)
        print(dir(y))

if __name__ == "__main__":
    test_model = TestModel()
    test_model.test_conv_segment()

