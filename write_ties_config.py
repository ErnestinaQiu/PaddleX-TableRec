"""
author: ErnestinaQiu
"""
import os
import yaml


def write_ties_config_file():
    """initiate, latter add in yml file directly
    """
    config = {}
    # the max vertices allowed
    config['max_vertices'] = 900  
    # the normalized shape
    config['normalized_width'] = 256
    config['normalized_height'] = 256

    # Vertex features (x, y, x2, y2, length of the word)
    config['num_vertex_features'] = 5
    # image shape in dataset
    config['image_height'] = 768
    config['image_width'] = 1366
    # the max number of words in a cell
    config['max_words_len'] = 30
    
    config['num_batch'] = 30
    # within the same cell, within the same row, within the same column, no relation
    config['num_global_features'] = 4
    # 1 for binary image, 3 for color image
    config['image_channels'] = 1
    # position
    config['dim_vertex_x_position'] = 0
    config['dim_vertex_y_position'] = 1
    config['dim_vertex_x2_position'] = 2
    config['dim_vertex_y2_position'] = 3
    # Number of vertices which are  actually in the sample
    # ??????????????????????????
    config['dim_num_vertices'] = 2

    # oversampling in ties monte carlo alg, but not used now
    config['samples_per_vertex'] = 6
    # model name in ties
    config['variable_scope'] = 'basic_conv_graph_alpha_1'
    # learning rate
    config['learning_rate'] = 0.0001
    # whether balanced labels in same cell, same col and same row, and the categories of table
    config['is_sampling_balanced'] = 1
    
    # momentum in Adam
    config['momentum'] = 0.6

    sp = os.path.join(os.getcwd(), 'exp', 'configs', 'ties.yml')
    with open(sp, 'w') as file:
        yaml.dump(config, file)


if __name__ == "__main__":
    write_ties_config_file()