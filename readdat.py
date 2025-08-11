"""
Title: Airfoil AutoEncoder
Author: tnkm23
Date created: 2025/08/12
Last modified: 2025/08/12
Description: Convolutional Variational AutoEncoder (VAE) trained on Airfoil shapes.
Accelerator: CPU

Airfoil data を読み込むモジュール
読み込んだAirfoilデータをVAEに入力するための前処理を行う

VAE
 入力：Airfoil形状データ
 出力：Airfoil形状データの再構成

"""

import numpy as np

def read_airfoil(data):
    """
    UIUCのairfoilデータを読み込む
    データの形式はXFLR5の.datと同じ
    データの最初の1行を読み飛ばす
    """

    # Read an airfoil in the same format as XFLR5 (.dat)
    with open(data, "r") as f:
        airfoil = f.readlines()
    # Split strings by tab
    airfoil = [line.split() for line in airfoil]
    # Convert strings in list to numpy array 
    airfoil = np.array(airfoil[1:],dtype='float64')

    return airfoil


if __name__ == "__main__":
    datapath = "UIUC Database"
    airfoil = read_airfoil(datapath + "/naca4412.dat")
    print(airfoil)
