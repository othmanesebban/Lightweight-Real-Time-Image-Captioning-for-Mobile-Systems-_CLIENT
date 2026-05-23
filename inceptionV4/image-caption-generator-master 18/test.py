import sys
from caption_generator import Caption_Generator  # Assuming your class is named Caption_Generator
from io import BytesIO
from PIL import Image

img_path = 'plane.jpg'

# Create a configuration object
class Config:
    def __init__(self, mode='test', other_parameters=None, dim_imgft=None, embedding_size=None, num_hidden=None
                 , batch_size=None, num_timesteps=None):
        self.mode = mode
        self.other_parameters = other_parameters  # Add other configuration parameters as needed
        self.dim_imgft = dim_imgft
        self.embedding_size = embedding_size
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_timesteps = num_timesteps
        self.num_timesteps = num_timesteps


# Before calling the Caption_Generator constructor
config = Config(mode='test', other_parameters={'param1': 'value1'}, dim_imgft=1536,
                embedding_size=256, num_hidden=512, batch_size=100 , num_timesteps=18)




print("Config:", config)

# Calling the Caption_Generator constructor
model = Caption_Generator(config=config)

# Check if a command-line argument is provided
if len(sys.argv) != 2:
    print("Usage: python your_script.py <image_path>")
    sys.exit(1)

# Use the command-line argument as the image path
image_path = sys.argv[1]

# Open the image file and read its content
with open(image_path, 'rb') as image_file:
    file = BytesIO(image_file.read())

img = Image.open(file)
# ... (previous code)

img.save(img_path)

decode_graph = model.build_decode_graph()

print("Debug: Before decoding")
print("Image Path:", img_path)
print("Decode Graph:", decode_graph)

caption = model.decode(decode_graph[0], decode_graph[1], img_path)

print("Debug: After decoding")
print("Caption:", caption)
