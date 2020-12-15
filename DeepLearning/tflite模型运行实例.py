import numpy as np
import tensorflow as tf

tt = np.loadtxt('test.txt')
interpreter = tf.lite.Interpreter(model_path="./model/model.tflite")
interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]
tt = tt.reshape((1,5000,1,1))
tt = tt.astype(np.float32)
interpreter.set_tensor(input_index, tt)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)
print(predictions)