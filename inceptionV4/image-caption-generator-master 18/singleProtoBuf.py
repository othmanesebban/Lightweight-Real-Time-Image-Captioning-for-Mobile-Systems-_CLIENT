import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
with open('model/Trained_Graphs/merged_frozen_graph.pb', 'rb') as f:
    fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='', op_dict=None, producer_op_list=None)
graph = tf.get_default_graph()
tensors = [n.name for n in tf.get_default_graph().as_graph_def().node]
wtoidx = np.load("Dataset/wordmap.npy",allow_pickle=True).tolist()
idxtow = dict(zip(wtoidx.values(), wtoidx.keys()))

with open("model/IdmapInceptionV3_RNN", 'w') as f:
    for value in idxtow.values():
        f.write(value + "\n")
print("Idmap is created")

