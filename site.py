import pickle

import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
import tempfile
import time

from PIL import Image

import happy

from happy import Cam



model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']


cap = cv2.VideoCapture(0)


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


labels_dict = {0: 'Hello', 1: 'How', 2: 'Are', 3: "You", 4: 'I', 5: 'I Am', 6: 'Ok', 7: 'Like', 8: 'A', 9: 'S', 10: 'L'}


char_list = []

char = ''


st.sidebar.title("Sidebar")
st.sidebar.subheader('Welcome')

app_mode = st.sidebar.selectbox('Choose App', 
                                ['Welcome', 'Translator', 'Learn']

                               )

if app_mode == 'Welcome':
    st.markdown('''
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 325px;
        }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 325px;
        }
        margin-left: -325px
    <style>
''', 
    unsafe_allow_html=True   
)
    st.title('Welcome')
    st.markdown('   Welcome! This application is designed to aid people around the world with disabilities who lead lives using **ASL**, and their families. Using artificial intelligence and landmark systems, we have created a way to learn and translate any ASL sign into words. (This project was created for the 2024 Science Fair Created By: Jheel Trivedi) ')
    img = Image.open("fox.png")

    st.image(
        img , 
        caption= "(Mascot made to help in the learning of children)" ,
        width= 650 ,
        channels= "RGB"
    )
    st.markdown('''
        To look at the digital **Write Up** please head to the link given below:

        - [Write Up (AI)](https://docs.google.com/document/d/1MzKAkzOtGgsnyGKDj6ThKigB2K0gMF8eWQ8D0DltBMA/edit)
         

        What does this site offer?

        - It is a hub of information regarding ASL and how to use it, the application provides many things and is meant to be a tool to be used at anytime in the day!
        - A translator that can breach the barriers of non speakers, it allows them to be understood even when the other person is obllivious to the language they are speaking.
        - I also belive learning is a big part of the world now, as it has for centuries, we provide a place where anyone can start learning ASL for free. These are structured with a more traditional style of flashcards, but also a visual AI combining modern technology to help them see what they learn.
        



        Thank you so much for visiting, I hope that this site will bring great learning opportunities to you all!
    
    
    ''')

elif app_mode == 'Translator':
    st.sidebar.markdown("----") 
    
    Turn_cam = st.sidebar.button("Use Camera")



    st.markdown('''
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 325px;
        }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 325px;
        }
        margin-left: -325px
    <style>
''', 
    unsafe_allow_html=True   
    )
    st.title("Traslator")
    st.markdown('''This is the translation page! Here using your **Webcam** we will be able to directly translate what you say into **ASL**!
                   To keep the expirence as accurate as possible, we have certain rules in place prior to starting up the translator, don't worry you can easily start it up again but these are just for a seamless expirence.

                   Rules:
                   - Make sure only one hand is in frame at a time
                   - Be careful with quick or odd hand positions
                   - Prepare yourself for a new world of  understanding


    
    
    ''')
    st.markdown("----------")
    
    frame_placeholder = st.empty()

    kpi1_text = st.markdown("")
    
    if Turn_cam:
        cap = cv2.VideoCapture(0)

        while cap.isOpened():


            data_aux = []
            x_ = []
            y_ = []

        
            ret, frame = cap.read()


            H, W, _ = frame.shape


            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
            
            



            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:


                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())


                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y


                        x_.append(x)
                        y_.append(y)


                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))


                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10


                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                try:
                    prediction = model.predict([np.asarray(data_aux)])
                except:
                    print("err")


                predicted_character = labels_dict[int(prediction[0])]


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                                cv2.LINE_AA)
                
                if len(char_list) == 0:
                    char_list.append(predicted_character)
                    char += predicted_character + " "

                else:
                    if char_list[-1] != predicted_character:
                        char_list.append(predicted_character)
                        char += predicted_character + " "

                # char = ' '.join(char_list)

                
            
        
            frame = cv2.resize(frame,(0,0),fx = 0.8 , fy = 0.8)

            #frame = image_resize(image = frame, width = 640)
            
            
            frame_placeholder.image(frame,channels = 'BGR',use_column_width=True)
            

            cv2.waitKey(1)
            
        
            kpi1_text.write(f"<h1 style='text-align: center; border-style:solid; padding: 5px;'>{char}</h1>", unsafe_allow_html=True)
            #color:rgb(255, 196, 146);
            time.sleep(0.1)




        cap.release()
        cv2.destroyAllWindows()



 
        
    st.markdown('---')

   




