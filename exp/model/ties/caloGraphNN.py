import paddle


def nearest_neighbor_matrix(spatial_features: paddle.Tensor, k=10):
    """Nearest neighbors matrix given spatial features.

    Args:
        spatial_features (paddle.Tensor): Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch, S = spatial features
        k (int, optional): Max neighbors. Defaults to 10.
    Returns:

    """
    

def indexing_tensor(spatial_features: paddle.Tensor, k=10, n_batch=-1):
    """

    Args:
        spatial_features (paddle.Tensor)
        k (int, optional): _description_. Defaults to 10.
        n_batch (int, optional): _description_. Defaults to -1.
    """
    shape_spatial_features = list(spatial_features.shape)
    n_batch = shape_spatial_features[0]
    n_max_entries = shape_spatial_features[1]

    # All of these tensors should be 3-dimensional
    assert len(shape_spatial_features) == 3

    # Neighbor matrix should be int as it should be used for indexing
    assert spatial_features.dtype == paddle.float32 or spatial_features.dtype == paddle.float64

    neighbor_matrix, distance_matrix = nearest_neighbor_matrix(spatial_features, k)

    batch_range = paddle.arange(0, n_batch).unsqueeze(1).unsqueeze(1).unsqueeze(1)



def high_dim_dense(inputs, nodes, **kwargs):
    """global transform to 3D

    Args:
        inputs (paddle.Tensor): 
        nodes (int): number of nodes

    Returns:
        _type_: _description_
    """
    if len(inputs.shape) == 3:
        con1d = paddle.nn.Conv1D(in_channels=inputs.shape[1], out_channels=nodes, kernel_size=(1), strides=1, padding=0, **kwargs)
        return con1d(inputs)

    if len(inputs.shape) == 4:
        con2d = paddle.nn.Conv2D(in_channels=inputs.shape[1], out_channels=nodes, kernel_size=(1, 1), strides=(1, 1), padding=0, **kwargs)
        return con2d(inputs)

    if len(inputs.shape) == 5:
        con3d = paddle.nn.Conv3D(in_channels=inputs.shape[1], out_channels=nodes, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding=0, **kwargs)
        return con3d(inputs)
