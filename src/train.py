import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping
from src.data_loader import split_dataset
from src.generators import create_datagen, multi_output_generator
from src.model_builder import build_model

train_files, val_files, _ = split_dataset()

train_gen = multi_output_generator(train_files, create_datagen(augment=True), batch_size=32)
val_gen = multi_output_generator(val_files, create_datagen(augment=False), batch_size=32, shuffle=False)

backbones = {
    "ResNet50": (ResNet50, 'avg'),
    "VGG16": (VGG16, 'flatten'),
    "EfficientNetB0": (EfficientNetB0, 'avg')
}

results = {}
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

for name, (model_class, pooling) in backbones.items():
    print(f"\n===== Training {name} =====")
    model = build_model(model_class, pooling)
    history = model.fit(train_gen, validation_data=val_gen, epochs=50, callbacks=[early_stop], verbose=1)
    results[name] = history.history
    pd.DataFrame(history.history).to_csv(f"{name}_history.csv", index=False)