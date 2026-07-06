from tensorflow.keras.preprocessing import image
import numpy as np
from NeuralNetwork import * 
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
CNN = createCNN()
CNN.load_weights('models/best_model.weights.h5')
img = load_and_preprocess_image('ISIC_0026622.jpg')
predict = CNN.predict(img)
predict_class = np.argmax(predict[0])
confidence = np.max(predict[0])
print(predict_class)
print(f"Уверенность: {confidence:.2%}")
