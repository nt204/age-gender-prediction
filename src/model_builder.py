from tensorflow.keras import layers, Model

def build_model(base_model_class, pooling_mode='avg'):
    if pooling_mode == 'flatten':
        base_model = base_model_class(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
        x = layers.Flatten()(base_model.output)
    else:
        base_model = base_model_class(include_top=False, input_shape=(128, 128, 3), pooling=pooling_mode, weights='imagenet')
        x = base_model.output
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    gender_output = layers.Dense(1, activation='sigmoid', name='gender_output')(x)
    age_output = layers.Dense(1, activation='linear', name='age_output')(x)
    model = Model(inputs=base_model.input, outputs=[gender_output, age_output])
    model.compile(
        optimizer='adam',
        loss={'gender_output': 'binary_crossentropy', 'age_output': 'mse'},
        metrics={'gender_output': 'accuracy', 'age_output': 'mae'}
    )
    return model
