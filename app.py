# import libraries
import streamlit as st
import numpy as np
import pandas as pd
import string
from PIL import Image
from tensorflow.keras.models import load_model

# set page configuration 
st.set_page_config(page_title='sign-language', layout='wide')

# load model and dataset pytho
@st.cache_resource
def load_cnn_model():
    return load_model('sign_lang.keras')
model= load_cnn_model()

train_df= pd.read_csv('sign_mnist_train.csv')
test_df= pd.read_csv('sign_mnist_test.csv')

# background image 
st.markdown(
    f"""
    <style>
    .stApp {{
    background-image: url("https://i.imgur.com/18dJu39.jpeg");
    background-size: cover;
    background-repeat: no-repeat;
    }}
    </style>""", unsafe_allow_html= True
)

# title
st.title("Hand Gesture Recognition Tasks(MNIST)")
tab1, tab2, tab3= st.tabs(['Dataset Info', 'Model Info', 'Predictions'])

# tab1 about dataset which is used.
with tab1:
    st.markdown(
        """
        <h2> About Dataset</h2>
        <p style='text-align:left; font-size: 18px;'>
            The sign Language MNIST dataset is a labeled images dataset designed for
            training machine learning and deep learning models to recognize American Sign LAnguage hand Gestures.
            The dataset contains images of size 28x28 pixels, where each image represents hand gesture
            corresponding to an ASL alphabet letter.
        </p>
        <h3>Data Split</h3> 
        <ul><li>Traing set: 27, 455 samples</li>
            <li>Test sets: 7, 172 samples</li>
        </ul>
        <h3>This dataset is commonly used for:</h3>
        <ul><li>Image Classification</li> 
            <li>Convolutional neural network training</li> 
            <li>hand gesture recognition</li>
            <li>Educational projects</li>
        </p>""", unsafe_allow_html= True
    )
    st.subheader('Dataset Review')
    with st.expander('Dataset Review'):
        st.write(train_df.head())

# tab2 about the model used 
with tab2:
    st.header('Convolutional Neural Network (CNN)')
    st.write('For train this data we used convolutional neural network. CNN is the advanced' \
    'version of the ANN, designed to extract features from grid like datasets. ' \
    'This is useful for the visual datasets such as images or videos.')
    st.write('CNN consist multiple layers like- Input Layer, Convolutional Layer, Pooling Layer, Activation layer,Flattening')
    st.info('1. Input Layer: In this layer we give input to our model. In CNN , the input will be an image or a sequence of images.' \
    'this layer holds raw input of image with width, height, depth.')
    st.info('2. Convolutional Layer: This layer extract features from the input dataset. IT applies set of filters' \
    'known as the kernals to input images. The kernals are small matrices 2x2, 3x3 shape. It slides over the image and find dot product between ' \
    'kernal and weights. The output of this layer refferd as feature maps')
    st.info('3. Activation Layer: Activation layer add nonlinearity to network. It apply activation functions Relu, tanh etc to the output of convolutional layer.')
    st.info('4. Pooling Layer: This layer reduce the size of feature map which make the computation fast , reduces memory and prevent ' \
    'overfitting. Tow common poolings are max pooling and averae pooling.')
    st.info('5. Flattening: The resulted feature maps are flattend into one-dimensional vector')


# tab3 prediction section
with tab3:
    labels= [c for c in string.ascii_uppercase if c not in ['J', 'Z']]
    st.title('Model Predictions')
    st.subheader('Upload a image to see the trained neural network prediction')

    uploaded_file= st.file_uploader(
        'Upload a image',
        type= ['png', 'jpg', 'jpeg']
    )
    if uploaded_file is not None:
        st.markdown('<h4 Upload Image </h4', unsafe_allow_html= True)
        st.image(uploaded_file, width=200)

        img= Image.open(uploaded_file).convert('L')
        img= img.resize((28,28))
        img_array= np.array(img)/ 255.0
        img_array= img_array.reshape(1, 28, 28, 1)

        prediction= model.predict(img_array)
        predicted_class= np.argmax(prediction)
        confidence= np.max(prediction)*100

        st.markdown(f"""
                    <div class= 'section'>
                    <h3> Prediction result</h3>
                    <p><b> Predicted class: </b> {labels[predicted_class]}<br>
                    <b> confidence: </b> {confidence:.2f}%
                    </p>
                    </div>
                    """, unsafe_allow_html= True)
