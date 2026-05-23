# file: keras2tensorflow.py
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import keras
from keras import backend as K
from keras.layers import GlobalAveragePooling2D, Input

tf.compat.v1.disable_eager_execution()

try:
    import keras.engine.saving as keras_saving

    _original_load_attributes_from_hdf5_group = keras_saving.load_attributes_from_hdf5_group

    def _patched_load_attributes_from_hdf5_group(group, name):
        data = _original_load_attributes_from_hdf5_group(group, name)
        return [item.decode("utf-8") if isinstance(item, bytes) else item for item in data]

    keras_saving.load_attributes_from_hdf5_group = _patched_load_attributes_from_hdf5_group
except Exception:
    pass


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    graph = session.graph
    with graph.as_default():
        keep_var_names = keep_var_names or []
        freeze_var_names = list(
            set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names)
        )

        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]

        input_graph_def = graph.as_graph_def()

        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""

        return tf.compat.v1.graph_util.convert_variables_to_constants(
            session,
            input_graph_def,
            output_names,
            freeze_var_names,
        )


def main():
    tf.compat.v1.keras.backend.set_learning_phase(0)

    base_model = keras.applications.InceptionV3(
        weights="imagenet",
        input_tensor=Input(shape=(299, 299, 3), name="input_1"),
        include_top=False,
    )

    output = GlobalAveragePooling2D(name="global_avg_pool")(base_model.output)
    model = keras.models.Model(
        inputs=base_model.inputs,
        outputs=output,
        name="inception_v3_gap",
    )

    model.summary()

    session = K.get_session()
    frozen_graph = freeze_session(
        session,
        output_names=[out.op.name for out in model.outputs],
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "ConvNets")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "inception_v3.pb")

    tf.io.write_graph(
        frozen_graph,
        output_dir,
        "inception_v3.pb",
        as_text=False,
    )

    print("input layer tensor:")
    for tensor in model.inputs:
        print(tensor.name)

    print("output layer tensor:")
    for tensor in model.outputs:
        print(tensor.name)

    print("they'll be used in convfeatures.py")
    print("CNN model is created!")
    print("Current working directory:", os.getcwd())
    print("Script directory:", script_dir)
    print("Expected .pb path:", output_path)
    print("File exists:", os.path.exists(output_path))

    if os.path.exists(output_path):
        print("File size (bytes):", os.path.getsize(output_path))
    else:
        raise FileNotFoundError("inception_v3.pb was not found after write_graph()")


if __name__ == "__main__":
    main()