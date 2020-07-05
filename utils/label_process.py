import numpy as np


def encode_labels(labels):

    """
    将标签图的灰度值转换成类别id
    注意：ignoreInEval为True的都当分类0处理
    @param labels: 标签灰度图
    """
    encoded_labels = np.zeros_like(labels)
    # 除了下面特意转换的，其余都属于类别0
    # 1
    encoded_labels[labels == 200] = 1
    encoded_labels[labels == 204] = 1
    encoded_labels[labels == 209] = 1
    # 2
    encoded_labels[labels == 201] = 2
    encoded_labels[labels == 203] = 2
    # 3
    encoded_labels[labels == 217] = 3
    # 4
    encoded_labels[labels == 210] = 4
    # 5
    encoded_labels[labels == 214] = 5
    # 6
    encoded_labels[labels == 220] = 6
    encoded_labels[labels == 221] = 6
    encoded_labels[labels == 222] = 6
    encoded_labels[labels == 224] = 6
    encoded_labels[labels == 225] = 6
    encoded_labels[labels == 226] = 6
    # 7
    encoded_labels[labels == 205] = 7
    encoded_labels[labels == 227] = 7
    encoded_labels[labels == 250] = 7
    return encoded_labels


def decode_labels(labels):

    decode_mask = np.zeros((labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[labels == 0] = 0
    # 1
    decode_mask[labels == 1] = 204
    # 2
    decode_mask[labels == 2] = 203
    # 3
    decode_mask[labels == 3] = 217
    # 4
    decode_mask[labels == 4] = 210
    # 5
    decode_mask[labels == 5] = 214
    # 6
    decode_mask[labels == 6] = 224
    # 7
    decode_mask[labels == 7] = 227

    return decode_mask


def decode_color_labels(labels):


    decode_mask = np.zeros((3, labels.shape[0], labels.shape[1]), dtype='uint8')
    # 0
    decode_mask[0][labels == 0] = 0
    decode_mask[1][labels == 0] = 0
    decode_mask[2][labels == 0] = 0
    # 1
    decode_mask[0][labels == 1] = 70
    decode_mask[1][labels == 1] = 130
    decode_mask[2][labels == 1] = 180
    # 2
    decode_mask[0][labels == 2] = 0
    decode_mask[1][labels == 2] = 0
    decode_mask[2][labels == 2] = 142
    # 3
    decode_mask[0][labels == 3] = 153
    decode_mask[1][labels == 3] = 153
    decode_mask[2][labels == 3] = 153
    # 4
    decode_mask[0][labels == 4] = 128
    decode_mask[1][labels == 4] = 64
    decode_mask[2][labels == 4] = 128
    # 5
    decode_mask[0][labels == 5] = 190
    decode_mask[1][labels == 5] = 153
    decode_mask[2][labels == 5] = 153
    # 6
    decode_mask[0][labels == 6] = 0
    decode_mask[1][labels == 6] = 0
    decode_mask[2][labels == 6] = 230
    # 7
    decode_mask[0][labels == 7] = 255
    decode_mask[1][labels == 7] = 128
    decode_mask[2][labels == 7] = 0

    return decode_mask


def verify_labels(labels):

    pixels = [0]
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            pixel = labels[x, y]
            if pixel not in pixels:
                pixels.append(pixel)
    print('The Labels Has Value:', pixels)