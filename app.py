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


@st.cache_resource
def algorithm(flag, mode="train"):
    if flag and mode == "train":
        solutions, map_pool, loss_log, world = train.learn()
        return solutions, map_pool, loss_log, world, flag

    else:
        return None, None, None, None, False


def import_parameters(container):
    df_hyper_p = pd.DataFrame(settings.HP_LIST, index=[1])
    edited_df = container.data_editor(df_hyper_p, num_rows="dynamic")

    if len(edited_df) > 0:
        settings.HP_LIST = edited_df.iloc[-1].to_dict()

    settings.NUM_EPOCHS = int(settings.HP_LIST["Episodes"])
    settings.TARGET_UPDATE = int(settings.HP_LIST["Target Update"])
    settings.BATCH_SIZE = int(settings.HP_LIST["Minibatch"])
    settings.REPLAY_SIZE = int(settings.HP_LIST["Replay Memory"])
    settings.EPSILON_START = settings.HP_LIST["Eps. Start"]
    settings.EPSILON_END = settings.HP_LIST["Eps. End"]
    settings.EPSILON_DECAY = settings.HP_LIST["Eps. Decay"]
    settings.GAMMA = settings.HP_LIST["Gamma"]
    settings.LEARNING_RATE = settings.HP_LIST["Learning Rate"]

def app():
    app_config()

    validation = st.session_state['validation']

    st.header('DRL - MDCVRP Dashboard')

    flag = st.session_state['flag']

    solutions, map_pool, loss_log, world, flag = algorithm(flag)

    col0_1, col0_2 = st.columns((1, 3))

    import_parameters(col0_2)

    if col0_1.button('Run training'):
        st.cache_resource.clear()
        st.session_state['validation'] = False
        solutions, map_pool, loss_log, world, flag = algorithm(True)

        st.session_state['flag'] = flag

    if flag:
        with st.expander("Analyze training solution pool"):
            ep_range_option = [i for i in range(len(solutions))]
            choice = st.selectbox("Select a solution you want to analyze", ep_range_option)

            depot = world.depots_coordinates[0]
            customers = world.customers_coordinates

            # Sample route (list of customer indices)
            routes = map_pool[choice]

            fig = go.Figure()

            # Plot depot
            fig.add_trace(go.Scatter(x=[depot[0]], y=[depot[1]], mode='markers', name='Depot',
                                     marker=dict(color='black', size=12, symbol='square')))

            # Plot customers
            customer_x = [c[0] for c in customers]
            customer_y = [c[1] for c in customers]
            fig.add_trace(go.Scatter(x=customer_x, y=customer_y, mode='markers', name='Customers',
                                     marker=dict(color='red', size=8)))

            # Plot routes
            for route in routes:
                route_coords = []
                for node in route:
                    if node == world.num_customers:
                        route_coords += [depot]
                    else:
                        route_coords += [customers[node]]

                route_x = [coord[0] for coord in route_coords]
                route_y = [coord[1] for coord in route_coords]
                fig.add_trace(go.Scatter(x=route_x, y=route_y, mode='lines+markers', name='Route'))

            fig.update_layout(title='Vehicle Routing Problem',
                              xaxis_title='X-coordinate',
                              yaxis_title='Y-coordinate')

            st.plotly_chart(fig)

        # plotting objective values
        df = pd.DataFrame({'Episodes': list(range(1, len(solutions) + 1)), 'Objective': solutions})
        fig = px.line(df, x='Episodes', y='Objective', title='Objective over episodes')
        st.plotly_chart(fig)

        # plotting loss values
        df = pd.DataFrame({'Episodes': list(range(1, len(loss_log) + 1)), 'Loss': loss_log})
        fig = px.line(df, x='Episodes', y='Loss', title='Loss over episodes')
        st.plotly_chart(fig)

if __name__ == '__main__':
    app()
