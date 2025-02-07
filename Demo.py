import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, concatenate, Input

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
    
    # Decoder (upsampling and concatenation with skip connections)
    x = base_model.output  # 16x16
    for skip in reversed(skips):
        x = UpSampling2D((2, 2))(x)
        x = concatenate([x, skip])
        x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)
    
    x = UpSampling2D((2, 2))(x)  # Final upsample to 512x512
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
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
