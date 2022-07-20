# preprocess_images_python_PIL_pillow

## Preprocess images (as numbers) using slim-resource python:
It is possible to preprocess images in python using just PIL/Pillow, the python image library.
e.g. Some image preprocessing using very resource and memory intensive (software development, NOT DEPLOYMENT) tools such as keras, whereas deployment requires slim tools such as TFlite and small python packages. 


#################################
# PIL/Pillow image preprocessing 
#################################

# load and resize image file
"""
equivilent of this from Keras:

img = image.load_img(img_path, target_size=(224, 224))
"""
img = Image.open(img_path)
img = img.resize((224, 224))

# image -> array
"""
equivilent of this from Keras:

img_array = image.img_to_array(img)
"""
img_array = np.asarray(img)

# already numpy
expanded_img_array = np.expand_dims(img_array, axis=0)

# already numpy
preprocessed_img = expanded_img_array / 255. 

# set: input_data = preprocessed image
input_data = preprocessed_img

# type cast to float32
input_data = input_data.astype('float32')
