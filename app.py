import streamlit as st
import train
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd
from PIL import Image
import numpy as np
from settings import NUM_EPOCHS, TARGET_UPDATE, DEVICE
import settings
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import model
import torch

def app_config():

    st.set_page_config(
        page_title="MDCVRP Planning",
        layout="wide",
        initial_sidebar_state="expanded",
        # menu_items={
        #    'Get Help': 'https://www.extremelycoolapp.com/help',
        #    'Report a bug': "https://www.extremelycoolapp.com/bug",
        #    'About': "# This is a header. This is an *extremely* cool app!"
        # }
    )

    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # Initialization
    if 'flag' not in st.session_state:
        st.session_state['flag'] = False
    if 'validation' not in st.session_state:
        st.session_state['validation'] = False


@st.cache_data
def algorithm(flag, mode="train"):
    if flag and mode == "train":
        solutions, loss_log = train.learn()
        return solutions, loss_log

    else:
        return None, None


def import_parameters(container):
    df_hyper_p = pd.DataFrame(settings.HP_LIST, index=[1])
    edited_df = container.data_editor(df_hyper_p, num_rows="dynamic")

    if len(edited_df) > 0:
        settings.HP_LIST = edited_df.iloc[-1].to_dict()

    settings.EPOCHS = int(settings.HP_LIST["Episodes"])
    settings.TARGET_UPDATE = int(settings.HP_LIST["Target Update"])
    settings.BATCH_SIZE = int(settings.HP_LIST["Minibatch"])
    settings.REPLAY_SIZE = int(settings.HP_LIST["Replay Memory"])
    settings.EPSILON_START = settings.HP_LIST["Eps. Start"]
    settings.EPSILON_END = settings.HP_LIST["Eps. End"]
    settings.EPSILON_DECAY = settings.HP_LIST["Eps. Decay"]
    settings.DISCOUNT = settings.HP_LIST["Discount"]
    settings.LEARNING_RATE = settings.HP_LIST["Learning Rate"]
    settings.GAMMA = settings.HP_LIST["LR Decay"]
    settings.SEED = int(settings.HP_LIST["Random Seed"])

def app():
    app_config()

    validation = st.session_state['validation']

    st.header('DRL - MDCVRP Dashboard')

    flag = st.session_state['flag']

    solutions, loss_log = algorithm(flag)

    col0_1, col0_2 = st.columns((1, 3))

    if col0_1.button('Run training'):
        st.cache_data.clear()
        st.session_state['validation'] = False
        solutions, loss_log = algorithm(True)

        st.session_state['flag'] = flag

        # plotting objective values
        df = pd.DataFrame({'Episodes': list(range(1, len(solutions) + 1)), 'Objective': solutions})
        fig = px.line(df, x='Episodes', y='Objective', title='Objective over episodes')
        st.plotly_chart(fig)

        # plotting loss values
        df = pd.DataFrame({'Episodes': list(range(1, len(loss_log) + 1)), 'Loss': loss_log})
        fig = px.line(df, x='Episodes', y='Loss', title='Loss over episodes')
        st.plotly_chart(fig)

    if flag:
        st.write(solutions)

if __name__ == '__main__':
    app()
