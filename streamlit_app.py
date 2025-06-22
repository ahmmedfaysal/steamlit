import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random

@st.cache_data
def load_mnist():
    (x_train, y_train), (_, _) = mnist.load_data()
    return x_train, y_train

x_train, y_train = load_mnist()

st.title("🖋️ Handwritten Digit Generator (0–9)")
st.write("Enter a digit from 0 to 9 and see 5 MNIST-style handwritten images.")

digit = st.number_input("Enter a digit (0–9):", min_value=0, max_value=9, step=1)

if st.button("Generate Images"):
    filtered_images = x_train[y_train == digit]

    selected_images = random.sample(list(filtered_images), 5)

    st.write(f"### Generated Images for Digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.image(selected_images[i], width=100, caption=f"{digit}", channels="L")

