# Lightweight Real-Time Image Captioning for Mobile Systems

This repository contains the Python training and model-export framework used in the paper:

**Lightweight Real-Time Image Captioning for Mobile Systems: An Optimized Multimodal Framework and Benchmarking Study**

The framework supports CNN-RNN image captioning models trained on MSCOCO 2017 and exported as frozen ProtoBuf models for Android deployment in the SeeAround mobile application.

## Supported Architectures

- InceptionV3-GRU
- ResNet50-GRU
- InceptionV4-LSTM

The optimized configuration reduces the maximum caption length from 22 to 18 tokens to improve deployment efficiency.

## Requirements

- Python 3.6
- TensorFlow 1.13.1
- Keras 2.2.4
- Joblib 1.0.1
- Matplotlib 3.3.4
- OpenCV 4.5.1
- Pandas 1.1.5
- NLTK 3.5
- MSCOCO 2017 images and captions

## Dataset

Download MSCOCO 2017 from:

https://cocodataset.org

Place the dataset as follows:

```text
Dataset/
 ├── captions_train2017.json
 ├── val2017/
 └── MSCOCO-images/

 ## Training Procedure
1. Clone the repository.
2. Place captions_train2017.json and MSCOCO 2017 images in the dataset folder.
3. Create the pre-trained TensorFlow graph:
python keras2tensorflow.py
4. Generate image features:
python convfeatures.py --data_path Dataset/MSCOCO-images --inception_path ConvNets/inception_v3.pb
5. Train the captioning model:
python main.py --mode train --caption_path ./Dataset/captions_train2017.json --feature_path ./Dataset/features.npy --resume
  ## Exporting the Model for Android Deployment
1. Save encoder and decoder graphs:
python main.py --mode test --image_path ANY_TEST_IMAGE.jpg --saveencoder --savedecoder
2. Freeze the encoder graph:
python utils/save_graph.py --mode encoder --model_folder model/Encoder/
3. Freeze the decoder graph:
python utils/save_graph.py --mode decoder --model_folder model/Decoder/
4. Merge encoder and decoder graphs:
python utils/merge_graphs.py --encpb ./model/Trained_Graphs/encoder_frozen_model.pb --decpb ./model/Trained_Graphs/decoder_frozen_model.pb
5. Generate the vocabulary/idmap file:
python utils/singleProtoBuf.py or utils/dualProtoBuf
  ## Evaluation
python main.py --mode eval --image_path Dataset/val2017 --inception_path ConvNets/inception_v3.pb
  ## Main Results Reported in the Paper
The optimized InceptionV4-LSTM configuration achieved:
| Metric                               | Value        |
| ------------------------------------ | ------------ |
| Inference time on Samsung Galaxy A11 | 178 ms/image |
| Throughput                           | 5.6 FPS      |
| Memory usage                         | 410 MB       |
| Composite operational score          | 92.6%        |

