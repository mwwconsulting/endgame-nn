# Import relevant modules
import numpy as np
import pandas as pd
import tensorflow as tf
import import_data as imp
import train_model as train
import define_model as defmod
import graph as gr
import gen_pos as gp


def ask_gen_training():
    yes_no = input("Do you need to generate endgame training data? ")
    if len(yes_no) == 0:
        pass
    elif yes_no[0].lower() == "y":
        gp.generate_training()


def ask_train_net():
    yes_no = input("Do you want to train a net? ")
    if len(yes_no) == 0:
        pass
    elif yes_no[0].lower() == "n":
        return

    # Get the data to use for training
    plane_version="v1"
    material_balance, target_count = gp.ask_for_input()
    input_file = "C:/games/chess/train_"+material_balance+str(int(target_count/1000))+"K"+plane_version+".npz"

    (x_train, y_train4) = imp.import_endgame(input_file)
    # Print a sample image
    print(x_train.shape)
    print(y_train4.shape)
    print(y_train4[18])
    y_train = y_train4[:, 3]  # Column 3 is the score (-2000, 2000)

    ##############
    # The following variables are the hyperparameters.
    learning_rate = 0.01
    epochs = 200
    batch_size = 2000
    validation_split = 0.2

    # Establish the model's topography.
    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="logs/bin2b/", histogram_freq=1)
    my_model = defmod.create_model_eg_bin2b(learning_rate)
    my_model.summary()
    # Train the model on the normalized training set.
    epochs, hist = train.train_model(my_model, tb_callback, x_train, y_train,
                                     epochs, batch_size, validation_split)
    print(hist.head())
    # Plot a graph of the metric vs. epochs.
    # list_of_metrics_to_plot = ['accuracy','val_accuracy']
    list_of_metrics_to_plot = ['loss', 'val_loss']
    gr.plot_curve(epochs, hist, list_of_metrics_to_plot)

    # print some of the predictions on train data compared to labels
    predictions = my_model.predict(x_train).flatten()
    print("ytrain predictions")
    diff = y_train - predictions
    for x in range(200):
        print(f"{y_train[x]} {predictions[x]:.3f} {int(diff[x])}")


def set_options():
    print(f"Using Tensorflow {tf.__version__}")
    # The following lines adjust the granularity of reporting.
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.6f}".format  # was .3f
    # The following line improves formatting when ouputting NumPy arrays.
    np.set_printoptions(linewidth=200)


if __name__ == '__main__':

    set_options()

    print("Welcome to chess_pos_gen")
    print("We are going to generate a file of chess endgame training examples")
    print("to use in training a neural net.")
    ask_gen_training()
    ask_train_net()

