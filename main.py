import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

NUM_BATCHES = 64
DIM_IMAGE = (256, 256)

# Load images from the directory "Dataset" into a format TensorFlow can use
image_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="Dataset",  
    batch_size=NUM_BATCHES,
    image_size=DIM_IMAGE
)
category_names = image_data.class_names
#TESTING IF WE'RE READING DATA CORRECTLY
# # Retrieve the names of the categories or classes of images
# category_names = image_data.class_names
# print(category_names)

# # Start creating a plot of size 10x10 inches to show images
# plt.figure(figsize=(10, 10))
# for img_batch, lbl_batch in image_data.take(1):  # Loop through one batch of images and their labels
#     for index in range(9):  # Display the first 9 images in this batch
#         ax = plt.subplot(3, 3, index + 1)  # Create a 3x3 grid for subplot
#         plt.imshow(img_batch[index].numpy().astype('uint8'))  # Display image
#         plt.title(category_names[lbl_batch[index]])  # Set title to the class name of the image
#         plt.axis("off")  # Hide the axes
# plt.show()  # Show the plot

# # Print the dimensions of the image and label batches for information
# for img_batch, lbl_batch in image_data:
#     print(f"Shape of image batch: {img_batch.shape}")
#     print(f"Shape of label batch: {lbl_batch.shape}")
#     break  # Stop after printing the first batch to avoid excessive output

# Define a function to partition the dataset into training, validation, and testing sets
def partition_dataset(data, split_train=0.8, split_val=0.1, split_test=0.1, enable_shuffle=True, shuffle_buffer=10000):
    assert (split_train + split_val + split_test) == 1  # Ensure splits add up to 100%
    
    size_dataset = len(data)  # Get total number of data points
    
    if enable_shuffle:  # If shuffling is enabled,
        data = data.shuffle(shuffle_buffer, seed=12)  # Shuffle the data
    
    size_train = int(split_train * size_dataset)  # Number of data points for training
    size_val = int(split_val * size_dataset)  # Number of data points for validation
    
    training_set = data.take(size_train)  # Get training data
    validation_set = data.skip(size_train).take(size_val)  # Get validation data
    testing_set = data.skip(size_train).skip(size_val)  # Get testing data
    
    return training_set, validation_set, testing_set

# Apply the function to partition the dataset
training_set, validation_set, testing_set = partition_dataset(image_data)

# Define the structure of a Convolutional Neural Network (CNN) model
num_labels = 4
cnn_model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),  # Normalize pixel values
  tf.keras.layers.Conv2D(32, 3, activation='relu'),  # Convolutional layer with 32 filters
  tf.keras.layers.MaxPooling2D(),  # Pooling layer to reduce dimensions
  tf.keras.layers.Conv2D(64, 3, activation='relu'),  # Another convolutional layer with 64 filters
  tf.keras.layers.MaxPooling2D(),  # Another pooling layer
  tf.keras.layers.Conv2D(64, 3, activation='relu'),  # Another convolutional layer
  tf.keras.layers.MaxPooling2D(),  # Final pooling layer
  tf.keras.layers.Flatten(),  # Flatten the output for the dense layer
  tf.keras.layers.Dense(128, activation='relu'),  # Dense layer with 128 units
  tf.keras.layers.Dense(num_labels)  # Output layer with units equal to the number of classes
])

# Compile the model specifying optimizer, loss function, and metrics to track
cnn_model.compile(
  optimizer='adam',
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

# Train the model on the training data, validate it on validation data
training_history = cnn_model.fit(
    training_set,
    batch_size=NUM_BATCHES,
    validation_data=validation_set,
    verbose=1,
    epochs=10,
)

# Evaluate the model on the testing data to see how well it does
eval_scores = cnn_model.evaluate(testing_set)

# Plot training and validation accuracy and loss
acc = training_history.history['accuracy']
val_acc = training_history.history['val_accuracy']
num_epochs = 10
loss = training_history.history['loss']
val_loss = training_history.history['val_loss']

plt.figure(figsize=(8, 8))  # Create a figure for plotting
plt.subplot(1, 2, 1)  # Subplot for accuracy
plt.plot(range(num_epochs), acc, label='Train Accuracy')
plt.plot(range(num_epochs), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)  # Subplot for loss
plt.plot(range(num_epochs), loss, label='Train Loss')
plt.plot(range(num_epochs), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss over Epochs')
plt.show()

# Define a function to predict the class of an image using the trained model
def predict_image_class(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)  # Convert image to array
    img_array = tf.expand_dims(img_array, 0)  # Add a batch dimension

    predictions = model.predict(img_array)  # Make predictions
    predicted_label = category_names[np.argmax(predictions[0])]  # Get the predicted class
    confidence = round(100 * np.max(predictions[0]), 2)  # Calculate confidence of the prediction
    return predicted_label, confidence

# Visualize predictions on one batch of the test set
plt.figure(figsize=(15, 15))  # Create a large figure
for img_batch, lbl_batch in testing_set.take(1):
    for idx in range(9):  # Display the first 9 images
        ax = plt.subplot(3, 3, idx + 1)
        if img_batch.shape[0] > idx:  # Ensure there is an image to display
            plt.imshow(img_batch[idx].numpy().astype("uint8"))  # Show the image
            try:
                predicted_label, confidence = predict_image_class(cnn_model, img_batch[idx])  # Predict the class
                actual_label = category_names[lbl_batch[idx]]  # Get the actual class
                print("\n")
                plt.title(f"Actual: {actual_label},\nPredicted: {predicted_label}.\nConfidence: {confidence}%")  # Display the prediction result
            except Exception as e:
                plt.title(f"Prediction error: {str(e)}")
            plt.axis("off")  # Turn off the axis
        else:
            plt.axis("off") 
plt.show() 