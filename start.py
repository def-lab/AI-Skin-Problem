from tensorflow.keras.preprocessing import image
import numpy as np
from NeuralNetwork import * 
import json
def numpy_encoder(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj
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
data={

    "predict":predict_class,
    "confidence":confidence
}
with open('query.json','w',encoding ='utf-8') as f:
    json.dump(data,f,default = numpy_encoder,ensure_ascii=False,indent=4)
print(predict_class)
print(f"Уверенность: {confidence:.2%}")
