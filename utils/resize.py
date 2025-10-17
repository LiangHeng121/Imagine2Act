import sys
from PIL import Image

input_path = sys.argv[1]
output_path = sys.argv[2]

image = Image.open(input_path)
resized_image = image.resize((512, 512))
resized_image.save(output_path)
