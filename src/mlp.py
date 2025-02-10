from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_mlp(input_shape, num_classes, topology=[512, 256, 128], dropout=False):
    if len(topology) < 2:
        print("Invalid topology. It must have at least 2 layers.")
        return None
    try:
        model = Sequential()
        
        model.add(Dense(topology[0], activation='relu', input_shape=input_shape))
        if dropout: model.add(Dropout(0.25))
        
        for units in topology[1:-1]:
            model.add(Dense(units, activation='relu'))
            if dropout: model.add(Dropout(0.25))
        
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