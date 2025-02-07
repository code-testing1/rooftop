import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, concatenate, Input, BatchNormalization, Activation

def upsample_block(x, skip, filters):
    """Upsampling block with Conv layers and skip connection"""
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)  # Use bilinear interpolation to match sizes
    x = concatenate([x, skip])  # Ensure shapes match before concatenation
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def unet_mobilenetv2(input_size=(512, 512, 3), num_classes=1):
    base_model = MobileNetV2(input_shape=input_size, include_top=False, weights="imagenet")
    
    # Extract encoder layers for skip connections
    skips = [
        base_model.get_layer("block_1_expand_relu").output,  # 256x256
        base_model.get_layer("block_3_expand_relu").output,  # 128x128
        base_model.get_layer("block_6_expand_relu").output,  # 64x64
        base_model.get_layer("block_13_expand_relu").output, # 32x32
        base_model.get_layer("block_16_project").output      # 16x16
    ]
    
    # Start decoding from the bottleneck (final encoder output)
    x = base_model.output  # 16x16
    x = Conv2D(512, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Decoder with corrected skip connections
    x = upsample_block(x, skips[4], 256)  # 32x32
    x = upsample_block(x, skips[3], 128)  # 64x64
    x = upsample_block(x, skips[2], 64)   # 128x128
    x = upsample_block(x, skips[1], 32)   # 256x256
    x = upsample_block(x, skips[0], 16)   # 512x512

    # Final segmentation output
    output_layer = Conv2D(num_classes, (1, 1), activation="sigmoid")(x)  # Binary segmentation

    model = Model(inputs=base_model.input, outputs=output_layer)
    
    return model

# Create model
model = unet_mobilenetv2()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]
)

# Print summary
model.summary()




import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Activation, concatenate, Input

def conv_block(x, filters):
    """Basic Convolutional Block: Conv2D -> BatchNorm -> ReLU"""
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def upsample_block(x, skip, filters):
    """Upsampling block with Transposed Convolution"""
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding="same")(x)  # Transposed Conv to upsample
    x = concatenate([x, skip])  # Ensure shapes match before concatenation
    x = conv_block(x, filters)
    return x

def unet_mobilenetv2(input_size=(512, 512, 3), num_classes=1):
    base_model = MobileNetV2(input_shape=input_size, include_top=False, weights="imagenet")

    # Extract encoder layers for skip connections
    skips = [
        base_model.get_layer("block_1_expand_relu").output,  # 256x256
        base_model.get_layer("block_3_expand_relu").output,  # 128x128
        base_model.get_layer("block_6_expand_relu").output,  # 64x64
        base_model.get_layer("block_13_expand_relu").output, # 32x32
        base_model.get_layer("block_16_project").output      # 16x16 (Bottleneck)
    ]

    # Start decoding from the bottleneck (final encoder output)
    x = skips[-1]  # 16x16 feature map
    x = conv_block(x, 512)  # Extra conv layer for better feature extraction

    # Decoder with proper upsampling and skip connections
    x = upsample_block(x, skips[3], 256)  # 32x32
    x = upsample_block(x, skips[2], 128)  # 64x64
    x = upsample_block(x, skips[1], 64)   # 128x128
    x = upsample_block(x, skips[0], 32)   # 256x256

    # Final upsampling to match the original image size
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(x)  # 512x512

    # Output layer
    output_layer = Conv2D(num_classes, (1, 1), activation="sigmoid")(x)  # Binary segmentation

    model = Model(inputs=base_model.input, outputs=output_layer)
    
    return model

# Create model
model = unet_mobilenetv2()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]
)

# Print summary
model.summary()

