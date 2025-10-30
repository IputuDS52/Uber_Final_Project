import streamlit as st
from PIL import Image

image_sidebar = Image.open('uber_banner.png')  # Replace with your image file
st.sidebar.image(image_sidebar, use_column_width=True)
st.sidebar.header('Tarif Uber')

image_banner = Image.open('uber_banner.png')  # Replace with your image file
st.image(image_banner, use_column_width=True)

from uber_ML import run_uber_ML

def main():
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox('Menu', menu)

    if choice == 'Home':
        st.subheader('Sedang dalam perjalanan ke tempat anda. kecepatan adalah prioritas kami')
    elif choice == 'Machine Learning':
        st.subheader('Estimasi tarif layanan uber')
        run_uber_ML()

if __name__ == '__main__':
    main()
