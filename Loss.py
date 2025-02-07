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
