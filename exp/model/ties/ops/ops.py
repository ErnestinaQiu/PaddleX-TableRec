"""
some operations for model
author: ErnestinaQiu
"""

def gather_features_from_conv_head(conv_head, vertices_y, vertices_x, vertices_y2, vertices_x2, scale_y, scale_x):
    """Gather features from a 2D image.

    Args:
        conv_head (_type_): The 2D conv head with shape [batch, height, width, channels]
        vertices_y (int): The y position of each of the vertex with shape [batch, max_vertices]
        vertices_x (int): The x position of each of the vertex with shape [batch, max_vertices]
        vertices_y2 (int): The height of each of the vertex with shape [batch, max_vertices]
        vertices_x2 (int): The width of each of the feature with shape [batch, max_vertices]
        scale_y (float): A scalar to show y_scale
        scale_x (float): A scalar to show x_scale
    Returns:
        features (): the gathered features with shape [batch, max_vertices, channels]
    """
    # normalization
    