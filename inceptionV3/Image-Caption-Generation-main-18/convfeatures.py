# file: convfeatures.py
import tensorflow as tf
import numpy as np
import argparse
import os

tf.compat.v1.disable_eager_execution()

batch_size = 10
files, input_layer, output_layer = [None] * 3

IMAGE_SIZE = 299
OUTPUT_SIZE = 2048


def build_prepro_graph(inception_path):
    global input_layer, output_layer

    with open(inception_path, "rb") as f:
        file_content = f.read()

    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(file_content)

    tf.import_graph_def(graph_def, name="import")
    graph = tf.compat.v1.get_default_graph()

    input_layer = graph.get_tensor_by_name("import/input_1:0")
    output_layer = graph.get_tensor_by_name("import/global_avg_pool/Mean:0")

    print("input_layer:")
    print(input_layer)
    print("output_layer:")
    print(output_layer)

    input_file = tf.compat.v1.placeholder(dtype=tf.string, name="InputFile")
    image_file = tf.io.read_file(input_file)

    jpg = tf.image.decode_jpeg(image_file, channels=3)
    png = tf.image.decode_png(image_file, channels=3)

    output_jpg = tf.image.resize(jpg, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
    output_jpg = tf.reshape(
        output_jpg,
        [1, IMAGE_SIZE, IMAGE_SIZE, 3],
        name="Preprocessed_JPG",
    )

    output_png = tf.image.resize(png, [IMAGE_SIZE, IMAGE_SIZE]) / 255.0
    output_png = tf.reshape(
        output_png,
        [1, IMAGE_SIZE, IMAGE_SIZE, 3],
        name="Preprocessed_PNG",
    )

    return input_file, output_jpg, output_png


def load_image(sess, io, image_path):
    if image_path.lower().endswith(".png"):
        return sess.run(io[2], feed_dict={io[0]: image_path})
    return sess.run(io[1], feed_dict={io[0]: image_path})


def load_next_batch(sess, io, img_path):
    global files

    for batch_idx in range(0, len(files), batch_size):
        batch_files = files[batch_idx:batch_idx + batch_size]
        batch_images = []

        for filename in batch_files:
            full_path = os.path.join(img_path, filename)
            batch_images.append(load_image(sess, io, full_path)[0])

        if len(batch_images) < batch_size:
            pad_count = batch_size - len(batch_images)
            batch_images.extend(
                [np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)] * pad_count
            )

        yield np.asarray(batch_images, dtype=np.float32), len(batch_files)


def forward_pass(io, img_path):
    global output_layer, files

    files = sorted(np.array(os.listdir(img_path)))
    print("#Images:", len(files))

    all_features = []

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        batch_iter = load_next_batch(sess, io, img_path)
        n_batch = int(np.ceil(len(files) / float(batch_size)))

        for i in range(n_batch):
            batch, valid_count = next(batch_iter)
            assert batch.shape == (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3)

            feed_dict = {input_layer: batch}
            batch_features = sess.run(output_layer, feed_dict=feed_dict).reshape(batch_size, OUTPUT_SIZE)

            all_features.append(batch_features[:valid_count])

            if i % 5 == 0 or i == n_batch - 1:
                progress = ((i + 1) / float(n_batch)) * 100.0
                print("Progress: {:.2f}%".format(progress))

    prob = np.concatenate(all_features, axis=0)

    os.makedirs("Dataset", exist_ok=True)
    np.save("Dataset/features.npy", prob)

    print("Saving Features: Dataset/features.npy")
    print("Saved shape:", prob.shape)


def get_features(sess, io, img, saveencoder=False):
    global output_layer

    reshaped_output = tf.reshape(output_layer, [1, OUTPUT_SIZE], name="Output_Features")
    image = load_image(sess, io, img)
    feed_dict = {input_layer: image}
    prob = sess.run(reshaped_output, feed_dict=feed_dict)

    if saveencoder:
        tensors = [n.name for n in sess.graph.as_graph_def().node]
        os.makedirs("model/Encoder", exist_ok=True)

        with open("model/Encoder/Encoder_Tensors.txt", "w", encoding="utf-8") as f:
            for tensor_name in tensors:
                f.write(tensor_name + "\n")

        saver = tf.compat.v1.train.Saver()
        saver.save(sess, "model/Encoder/model.ckpt")

    return prob


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to images folder",
        required=True,
    )
    parser.add_argument(
        "--inception_path",
        type=str,
        help="Path to inception_v3.pb",
        required=True,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    print("Extracting Features")
    io = build_prepro_graph(args.inception_path)
    forward_pass(io, args.data_path)
    print("done")