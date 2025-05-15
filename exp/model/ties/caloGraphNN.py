import paddle


def euclidean_squared(A, B):
    """Returns euclidean distance between two batches of shape [B,N,F] and [B,M,F] where B is batch size, N is number of
    examples in the batch of first set, M is number of examples in the batch of second set, F is number of spatial
    features.

    Args:
        A (paddle.Tensor): shape [B,N,F]
        B (paddle.Tensor): shape [B,M,F]
    Returns:
        A matrix of size [B, N, M] where each element [i,j] denotes euclidean distance between ith entry in first set and
        jth in second set.
    """
    shape_A = list(A.shape)
    shape_B = list(B.shape)

    assert (A.dtype == paddle.float32 or A.dtype == paddle.float64) and (B.dtype == paddle.float32 or B.dtype == paddle.float64)
    assert len(shape_A) == 3 and len(shape_B) == 3
    assert shape_A[0] == shape_B[0]

    # Finds euclidean distance using property (a-b)^2 = a^2 + b^2 - 2ab
    sub_factor = -2 * paddle.matmul(A, paddle.transpose(B, perm=(0, 2, 1)))
    dotA = paddle.unsqueeze(paddle.sum(A * A, axis=2), axis=2)
    dotB = paddle.unsqueeze(paddle.sum(B * B, axis=2), axis=1)
    return paddle.abs(sub_factor + dotA + dotB)


def nearest_neighbor_matrix(spatial_features: paddle.Tensor, k=10):
    """Nearest neighbors matrix given spatial features.

    Args:
        spatial_features (paddle.Tensor): Spatial features of shape [B, N, S] where B = batch size, N = max examples in batch, S = spatial features
        k (int, optional): Max neighbors. Defaults to 10.
    Returns:
        N (paddle.Tensor): neighbor index matrix
        D (paddle.Tensor): distance matrix
    """
    shape = list(spatial_features.shape)

    assert spatial_features.dtype == paddle.float32 or spatial_features.dtype == paddle.float64
    assert len(shape) == 3

    D = euclidean_squared(spatial_features, spatial_features)
    D, N = paddle.topk(-D, k)
    return N, -D


def indexing_tensor(spatial_features: paddle.Tensor, k=10):
    """

    Args:
        spatial_features (paddle.Tensor)
        k (int, optional): the k top nearest nodes. Defaults to 10.
    Returns:
        indexing_tensor (paddle.Tensor): the top k nearest neighbors of each vertexes
        distance_matrix (paddle.Tensor): the euclidean distance of spatial features
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
    batch_range = paddle.tile(batch_range, (1, n_max_entries, k, 1))
    expanded_neighbor_matrix = neighbor_matrix.unsqueeze(3)

    indexing_tensor = paddle.concat([batch_range, expanded_neighbor_matrix], axis=3)

    return indexing_tensor.to(paddle.int64), distance_matrix


def high_dim_dense(inputs, nodes, **kwargs):
    """use convolution to global transform into 3D

    Args:
        inputs (paddle.Tensor): tensor
        nodes (int): number of nodes

    Returns:
        _type_: _description_
    """
    if len(inputs.shape) == 3:
        con1d = paddle.nn.Conv1D(in_channels=inputs.shape[1], out_channels=nodes, kernel_size=(1), stride=1, padding=0, **kwargs)
        return con1d(inputs)

    if len(inputs.shape) == 4:
        con2d = paddle.nn.Conv2D(in_channels=inputs.shape[1], out_channels=nodes, kernel_size=(1, 1), stride=(1, 1), padding=0, **kwargs)
        return con2d(inputs)

    if len(inputs.shape) == 5:
        con3d = paddle.nn.Conv3D(in_channels=inputs.shape[1], out_channels=nodes, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0, **kwargs)
        return con3d(inputs)


def layer_global_exchange(vertices_in):
    trans_vertices_in = vertices_in

    global_summed = paddle.mean(trans_vertices_in, axis=1, keepdim=True)

    global_summed = paddle.tile(global_summed, [1, vertices_in.shape[1], 1])
    vertices_out = paddle.concat([vertices_in, global_summed], axis=-1)

    return vertices_out


