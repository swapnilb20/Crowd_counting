from src.dataset.tf_dataset import make_dataset
from src.models.convnext_csrnet import build_convnext_crowd
from src.models.loss_metrics import combined_loss, count_mae_metric
from src.utils.visualize import visualize_predictions
import tensorflow as tf

train = make_dataset("data/train/img", "data/densities/train", 16)
val   = make_dataset("data/val/img", "data/densities/val", 16)
test  = make_dataset("data/test/img", "data/densities/test", 16)

model = build_convnext_crowd()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=combined_loss, metrics=[count_mae_metric]
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("crowd_model.keras", save_best_only=True, monitor="val_count_mae_metric", mode="min"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_count_mae_metric", factor=0.5, patience=3),
    tf.keras.callbacks.EarlyStopping(monitor="val_count_mae_metric", patience=15, restore_best_weights=True)
]

model.fit(train, validation_data=val, epochs=50, callbacks=callbacks)
print(model.evaluate(test))
visualize_predictions(model, test)
