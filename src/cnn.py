import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def plot_first_images(X, y, num_images=5):
    
    plt.figure(figsize=(10, 5))

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(X[i].squeeze(), cmap="gray")  # Remove extra dimension for visualization
        plt.title(f"Label: {y[i]}")  # Show label
        plt.axis("off")

    plt.show()


def build_cnn(input_shape, num_classes, topology=[32, 64, 256], dropout=False):
    if len(topology) < 3:
        print("Invalid topology. It must have at least 3 layers.")
        return None
    try:
        model = Sequential()

        model.add(Conv2D(topology[0], kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout: model.add(Dropout(0.25))

        for filters in topology[1:-2]:
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            if dropout: model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(topology[-1], activation='relu'))
        if dropout: model.add(Dropout(0.5))

        model.add(Dense(num_classes, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print("Model created successfully!")
        model.summary()
    except Exception as e:
        print("Error creating model: ",e)
        return None
    
    return model