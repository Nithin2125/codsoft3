import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained ResNet50 model
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = resnet_model.predict(img_array)
    return features

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_caption(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        max_length=50,
        num_return_sequences=1
    )
    caption = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return caption

def generate_caption_for_image(img_path):
    # Extract image features
    image_features = extract_features(img_path)
    
    # Example prompt based on image features
    prompt = "A photo of"
    
    # Generate caption using GPT-2
    caption = generate_caption(prompt)
    
    return caption

# Example usage
img_path = 'path/to/your/image.jpg'
caption = generate_caption_for_image(img_path)
print(caption)
