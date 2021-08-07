import tensorflow as tf

def create_model_eg1(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8, 15)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model

def create_model_eg2(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8, 15)))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model


def create_model_eg2b(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8, 15)))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model


def create_model_eg2c(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8, 15)))
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model


def create_model_eg3(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model

def create_model_eg3b(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model

def create_model_eg3c(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=1))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=['MeanSquaredError'])

    return model

def create_model_eg_bin1(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(8, 8, 15)))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=32, activation='relu'))
    model.add(tf.keras.layers.Dense(units=33))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model

def create_model_eg_bin2(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=33))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def create_model_eg_bin2b(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3,3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=33))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model


def create_model_eg_bin2c(my_learning_rate):
    """Create and compile a deep neural net."""
    # This is a first try to get a simple model that works
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3,3), input_shape=(8,8,15), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=(3,3), strides=(1, 1), padding='same'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=33))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    return model
