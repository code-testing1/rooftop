import tensorflow.keras.backend as K

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth))

# Combined BCE + Dice Loss
def bce_dice_loss(y_true, y_pred):
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)

# Compile with new loss function
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=bce_dice_loss,
    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)]
)


# Count the number of roof pixels vs. background pixels
total_pixels = np.prod(Y_train.shape)
roof_pixels = np.sum(Y_train)
background_pixels = total_pixels - roof_pixels

print(f"Roof Pixels: {roof_pixels}, Background Pixels: {background_pixels}")

#if imbalance too high
roof_weight = background_pixels / total_pixels
bg_weight = roof_pixels / total_pixels

loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
weighted_loss = lambda y_true, y_pred: loss(y_true, y_pred) * (y_true * roof_weight + (1 - y_true) * bg_weight)


optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)  # Reduce LR
model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
