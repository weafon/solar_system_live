import tensorflow as tf

import numpy as np

# ... (Your model creation and training code) ...
#model = tf.keras.layers.TFSMLayer(cycle60.karas, call_endpoint='serving_default')
model = tf.keras.models.load_model("cycle60.h5") # Loads from the directory

# Get the first layer (assuming it's a Dense layer)
first_layer = model.layers[0]  # Layers are indexed from 0

# Get the weights and biases
weights = first_layer.get_weights()[0]  # Weights are at index 0
biases = first_layer.get_weights()[1]   # Biases are at index 1

print("Weights shape:", weights.shape)
print("Biases shape:", biases.shape)

# Example: Print the weights (or save them to a file)
print("Weights:\n", weights)

# Example: Save weights to a text file (e.g., for inspection or other tools)
np.savetxt("first_layer_weights.txt", weights)
np.savetxt("first_layer_biases.txt", biases)