"""
author: ErnestinaQiu
"""
import paddle
import numpy as np
from exp.model.ties.models.conv_segment import BasicConvSegment
from exp.model.ties.models.dgcnn_segment import Dgcnn
from exp.model.ties.ops.ties import indexing_tensor
from exp.model.ties.caloGraphNN import high_dim_dense


class TestModel:
    def test_conv_segment(self):
        conv = BasicConvSegment()
        x = np.ones(shape=(1, 1, 700, 100))
        y = conv(x)
        print(y)
        print(y.shape)
        print(dir(y))

    def test_indexing_tensor(self):
        x = paddle.ones(shape=(2, 3, 4))
        indexing_matrix, distance_matrix = indexing_tensor(spatial_features=x, k=10, n_batch=-1)
        print(f"indexing_matrix.shape: {indexing_matrix.shape}, distance_matrix.shape: {distance_matrix.shape}")

    def test_high_dim_dense(self):
        x = paddle.ones(shape=(2, 3, 4, 5))
        y = high_dim_dense(x, 64)
        print(y.shape)

if __name__ == "__main__":
    test_model = TestModel()
    # test_model.test_conv_segment()
    # test_model.test_indexing_tensor()
    test_model.test_high_dim_dense()
