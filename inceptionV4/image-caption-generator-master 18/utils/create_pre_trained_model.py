import tensorflow as tf

graph_file = "model/Trained_Graphs/merged_frozen_InceptionV4-80.pb"
graph_def = tf.GraphDef()
with open(graph_file, "rb") as f:
  graph_def.ParseFromString(f.read())

for node in graph_def.node:
  print(node.name)