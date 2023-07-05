import os
import pandas as pd
from datasets import load_dataset
import streamlit as st

PATH_TO_DATA = 'data'


def get_first_file_with_pattern(path, pattern):
    files = os.listdir(path)
    files = [f for f in files if pattern in f]
    return files[0]


@st.cache_data(ttl=3600)
def load_transaction_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'tx_')
    df = pd.read_parquet(os.path.join(PATH_TO_DATA, file))
    return df


@st.cache_data(ttl=3600)
def load_votes_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'votes_')
    df = pd.read_csv(os.path.join(PATH_TO_DATA, file))
    return df


@st.cache_data(ttl=3600)
def load_projects_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'projects_')
    df = pd.read_csv(os.path.join(PATH_TO_DATA, file))
    return df


@st.cache_data(ttl=3600)
def load_features_voters_data():
    file = get_first_file_with_pattern(PATH_TO_DATA, 'voters_features_')
    df = pd.read_csv(os.path.join(PATH_TO_DATA, file))
    return df


def load_train_data_hug(path):
    ds = load_dataset(path=path)
    df = ds['train'].to_pandas()
    return df


@st.cache_data(ttl=3600)
def load_transaction_data_hug():
    return load_train_data_hug('Poupou/citizen-round-transactions')


@st.cache_data(ttl=3600)
def load_votes_data_hug():
    return load_train_data_hug('Poupou/citizen-round-votes')


@st.cache_data(ttl=3600)
def load_features_voters_data_hug():
    return load_train_data_hug('Poupou/citizen-round-features')
