import sys, os
from caption_generator import *
from configuration import Configuration

args= {"mode": "test", "resume":0, "load_image":1,"saveencoder":0, "savedecoder":0}
config =Configuration(args)
model = Caption_Generator(config)

#print(model)
#print(config)

path="Images/"
files=sorted(os.listdir(path))
files=[path+f for f in files]

for f in files:
    if os.path.splitext(f)[1] in [".png",".jpg",".jpeg"]:
        model.decode(f)