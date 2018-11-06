import pickle
import json
import numpy
from sklearn.externals import joblib
from sklearn.linear_model import Ridge
from azureml.core.model import Model
import tensorflow as tf

def init():
    global model
    global sess
    tf.reset_default_graph()
    # note here "sklearn_regression_model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('ml-model') + "/"
    saver =  tf.train.import_meta_graph(model_path + "mnist-tf.model.meta")
    graph = tf.get_default_graph()
    sess = tf.Session()
    saver.restore(sess, model_path + "mnist-tf.model")
    global w,b
    w, b = sess.run(["W:0", "B:0"])

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        data = data.reshape(1, 13)
        result = sess.run(tf.add(b, tf.matmul(data.reshape(1,13), w)))
        result = result[0][0]
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result})



'''
def init():
    global model
    # note here "sklearn_regression_model.pkl" is the name of the model registered under
    # this is a different behavior than before when the code is run locally, even though the code is the same.
    model_path = Model.get_model_path('ml-model')
    # deserialize the model file back into a sklearn model
    model = joblib.load(model_path)

# note you can pass in multiple rows for scoring
def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = numpy.array(data)
        result = model.predict(data)
    except Exception as e:
        result = str(e)
    return json.dumps({"result": result.tolist()})
    '''
