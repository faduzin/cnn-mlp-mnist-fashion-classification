import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print('Data loaded successfully.')
        return data
    except FileNotFoundError:
        print('File not found.')
        return None
    

def save_data(df, file_path):
    try: # Tenta salvar o arquivo
        df.to_csv(file_path, index=False) # Salva o dataframe
        print(f"Arquivo salvo em: {file_path}.") # Exibe mensagem de sucesso
    except: # Se houver erro
        raise("Falha ao salvar arquivo.") # Exibe mensagem de erro


def data_info(data):
    print(data.info())
    print("-" * 50)
    print(data.describe())
    print("-" * 50)
    print(data.head(5))
    print("-" * 50)
    print("Data shape: ",data.shape)
    print("Amount of duplicates: ", data.duplicated().sum())


def count_classes(y):
    unique_labels, counts = np.unique(y, return_counts=True)

    # Display the results
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} occurrences")


def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Acurácia de Treinamento')
    plt.plot(epochs, val_acc, 'ro-', label='Acurácia de Validação')
    plt.title('Acurácia de Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Perda de Treinamento')
    plt.plot(epochs, val_loss, 'ro-', label='Perda de Validação')
    plt.title('Perda de Treinamento e Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Perda')
    plt.legend()

    plt.tight_layout()
    plt.show()


def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
