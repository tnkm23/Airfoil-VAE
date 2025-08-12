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
        
    coords = []
    for line in airfoil[1:]:  # 1行目はヘッダー
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
            except ValueError:
                continue  # 数値変換できない行はスキップ
        else:
            continue  # 空行や座標以外の行はスキップ
    if len(coords) == 0:
        return np.empty((0,2), dtype='float64')  # 座標がない場合は空配列
    return np.array(coords, dtype='float64')




if __name__ == "__main__":
    datapath = "UIUC Database"
    airfoil = read_airfoil(datapath + "/naca4412.dat")
    print(airfoil)
