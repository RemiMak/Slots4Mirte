import tensorflow as tf
import keras
import numpy as np

def get_mobilenet_backbone(image_shape: tuple[int, int]):
    assert image_shape[0] >= 32 and image_shape[1] >= 32 # requirement set by docs

    og_mobilenet = keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape= (*image_shape, 3),
    )
 # just the kernel
    modified_mobilenet = keras.applications.MobileNetV2(
        weights=None,
        include_top=False,
        input_shape= (*image_shape, 4),
    )
    
    # find the first conv2D layer
    first_conv_layer = None
    first_conv_idx = -1
    
    for i, layer in enumerate(og_mobilenet.layers):
        if isinstance(layer, keras.layers.Conv2D):
            first_conv_layer = layer
            first_conv_idx = i
            break

    assert first_conv_idx != -1 and first_conv_layer is not None, "Could not find first conv2D layer"

    # Note: dimensions of conv2D layers weights are: height x width x input channels x output channels

    # calculate the depth kernel as the mean of the rgb kernels
    rgb_kernel = first_conv_layer.get_weights()[0]
    depth_kernel = np.mean(rgb_kernel, axis=2, keepdims=True)

    # copy rgb_kernels and concatenate depth kernel
    rgbd_kernel = np.concatenate([rgb_kernel, depth_kernel], axis=2)
    modified_mobilenet.layers[first_conv_idx].set_weights([rgbd_kernel] + first_conv_layer.get_weights()[1:])

    # copy all other weights
    for i in range(first_conv_idx + 1, len(og_mobilenet.layers)):
        og_weights = og_mobilenet.layers[i].get_weights()
        if og_weights:
            modified_mobilenet.layers[i].set_weights(og_weights)

    return modified_mobilenet