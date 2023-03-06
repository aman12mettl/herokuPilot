from PIL import Image


img = Image.open("static/placeholder2.jpg")

img = img.resize((320,240))
img.save("resized.jpg")