import requests
import io
import numpy as np
from PIL import Image
url = "http://0.0.0.0:8000/predict_image/"

#fin = open('file_1.log', 'rb')
fin = open('test_image.png', 'rb')
files = {'file': fin}
r = requests.post(url, files=files)

#text = open(io.BytesIO(fin), "r").read()

print(np.array(Image.open(fin)).shape)

fin.close()
