import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ... (Load and preprocess your data, X: 6-dim inputs, y: labels) ...

import numpy as np

def load_data(filepath):
    """Loads data from a text file.

    Args:
        filepath: The path to the text file.

    Returns:
        X: A NumPy array of input features (6 dimensions).
        y: A NumPy array of labels (0 or 1).
        or None, None if there's an error during the load.
    """
    try:
        data = np.loadtxt(filepath)  # Efficiently load numeric data
        X = data[:, :-1]  # All columns except the last one (features)
        y = data[:, -1].astype(int) # The last column (labels), convert to integers
        return X, y
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except ValueError: # handle potential errors such as non-numeric data in file.
        print(f"Error: Invalid data format in {filepath}")
        return None, None
    except Exception as e: # Catch any other potential exceptions
        print(f"An unexpected error occurred: {e}")
        return None, None


# Example usage:
filepath = "train.txt"  # Replace with the actual path to your file
X, y = load_data(filepath)

if X is not None and y is not None:
    print("Data loaded successfully:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First few rows of X:\n", X[:5])  # Print the first 5 rows
    print("First few values of y:\n", y[:5])
    # Now you can use X and y for training your neural network.


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1765, random_state=42) # 0.15/0.85

# Scale data (important!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(8,)), # Input layer and first hidden layer
    tf.keras.layers.Dense(32, activation='relu'),                 # Second hidden layer (example)
    tf.keras.layers.Dense(1, activation='sigmoid')                # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))
model.save("cycle60.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# ... (Make predictions on new data) ...