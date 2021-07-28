import os
import pandas as pd


def load_data(path):

    d = pd.read_csv(path)

    d = d.drop(['Usage', 'NF'], axis=1)

    return d["Image name"].to_numpy(), d.iloc[:, 2:].to_numpy()/10


def mk_dir(dir):
    try:
        os.mkdir( dir )
    except OSError:
        pass
