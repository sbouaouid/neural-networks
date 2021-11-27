import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import matplotlib.pyplot as plt 

#importer les données de Fashion-MNIST 
data = keras.datasets.fashion_mnist 

#ici on est en train de générer des données mais d'une manière séparée
#c'est à dire que les données sont divisées sur deux parties
#la première pour l'entrainement 
#la deuxième pour le test 
(train_images, train_labels), (test_images, test_labels)= data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#print(train_labels[5])
#print(class_names[4]) 

print(train_images[7])

#après avoir tester la commande ci-dessus, nous avons remarqué 
#que les chiffres composants l'image sont très grands
#et du coup nous avons décidé de les diviser sur 255.0
train_images = train_images/255.0
test_images = test_images/255.0

#print(train_images[7])

#plt.imshow(train_images[7], cmap=plt.cm.binary)
#plt.show()

model = keras.Sequential([
    # la première couche, où on passe l'ensemble de nos données 
    keras.layers.Flatten(input_shape=(28,28)), # liste de 784 valeurs entre 0 et 1
    # la couche cachée
    keras.layers.Dense(128, activation="relu"),
    # la couche sortie 
    keras.layers.Dense(10, activation="softmax")   
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_images, train_labels, epochs = 5)

prediction = model.predict(test_images)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap = plt.cm.binary)
    plt.xlabel("Actual : " + class_names[test_labels[i]])
    plt.title("Prediction : " + class_names[np.argmax(prediction[i])])
    plt.show()