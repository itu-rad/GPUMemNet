import requests
import os
from utils import read_yaml


def get_transformers_data(save_dir):
    url = "https://www.dropbox.com/scl/fi/24nzzwqm7rkhhjaaugeb9/transformer_data1.csv?rlkey=bo3jf4ee5iuhpjlbf339mp46v&st=6djbnkp4&dl=1"
    response = requests.get(url)

    # Save the file locally
    with open(os.path.join(save_dir, 'trans_data.csv'), 'wb') as file:
        file.write(response.content)


def get_cnn_data(save_dir):
    url = "https://www.dropbox.com/scl/fi/x2eh2yil56rgu5id1fwbm/cnn_data1.csv?rlkey=7bltxkitlwdnuirw08r2ysuv4&st=5z04ff8s&dl=1"
    response = requests.get(url)

    # Save the file locally
    with open(os.path.join(save_dir,'cnn_data1.csv'), 'wb') as file:
        file.write(response.content)

def get_mlp_data(save_dir):
    url = "https://www.dropbox.com/scl/fi/wxc1jz8spd0thbzi4oj3h/mlp_data2.csv?rlkey=p0a22wgqlxdnxm7pum27lh86y&st=jelhn8kg&dl=1"
    response = requests.get(url)

    # Save the file locally
    with open(os.path.join(save_dir,'mlp_data2.csv'), 'wb') as file:
        file.write(response.content)


if __name__=="__main__":
    config = read_yaml("config.yaml")
    save_dir = config["data_dir"]

    get_transformers_data(save_dir)
    get_cnn_data(save_dir)
    get_mlp_data(save_dir)