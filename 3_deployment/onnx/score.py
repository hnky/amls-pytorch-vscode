import os
import json
import requests
from io import BytesIO
import time
import numpy as np
from PIL import Image
import onnxruntime

def init():
    global ort_session
    global labels

    model_file = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'simpsons-classification-onnx','model.onnx')
    model_labels = os.path.join(os.getenv('AZUREML_MODEL_DIR'),'simpsons-classification-onnx', 'labels.txt')
    
    print('Loading model...', end='')
    ort_session = onnxruntime.InferenceSession(model_file)

    print('Loading labels...', end='')
    labels = load_labels(model_labels)
    print(len(labels), 'found. Success!')

def run(input_data):
    url = json.loads(input_data)['image']

    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    input_data = preprocess(image)

    input_name = ort_session.get_inputs()[0].name  
   
    start = time.time()
    raw_result = ort_session.run([], {input_name: input_data})
    end = time.time()

    res = postprocess(raw_result)
    idx = np.argmax(res)

    inference_time = np.round((end - start) * 1000, 2)

    result = {
        'time': str(inference_time)+'ms',
        'prediction': labels[idx],
        'confidence': "{:.2f}".format(res[idx]*100)+"%"
    }

    return result


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def load_labels(path):
    with open(path) as lf:
        labels = [l.strip() for l in lf.readlines()]
    return np.asarray(labels)

def preprocess(input_image):
    # Resize image
    input_image = input_image.resize((224,224)) 

    # Convert from RGB to BGR
    input_data = np.array(input_image).transpose(2, 0, 1)

    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
  
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()