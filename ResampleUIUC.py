"""
UIUCのairfoilデータを読み込み、正規化とリサンプリングを行うモジュール
"""

import numpy as np
import glob
from scipy import interpolate
import os

def load_airfoil_dat(file_path):
    """UIUCのdatファイルを読み込んでnumpy配列に変換"""
    coords = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                x, y = map(float, line.strip().split())
                coords.append([x, y])
            except ValueError:
                # ヘッダ行や不正行はスキップ
                continue
    coords = np.array(coords)
    if coords.shape[0] < 10:
        raise ValueError("点が少なすぎる")
    return coords

def normalize_and_resample(coords, n_points=200):
    """翼弦長を1に正規化し、スプライン補間で等間隔サンプリング"""
    # 正規化（x座標の最小を0に、最大を1に）
    coords[:, 0] -= coords[:, 0].min()
    coords /= coords[:, 0].max()

    # 曲線長に沿ったパラメータ
    dist = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)  # 最初に0を追加
    t = dist / dist[-1]

    # スプライン補間
    fx = interpolate.interp1d(t, coords[:, 0])
    fy = interpolate.interp1d(t, coords[:, 1])

    # 等間隔サンプリング
    t_new = np.linspace(0, 1, n_points)
    x_new = fx(t_new)
    y_new = fy(t_new)

    return np.stack([x_new, y_new], axis=1)


def normalize_and_resample_uiuc(coords, n_points=200):
    # 翼弦長で正規化
    coords[:, 0] -= coords[:, 0].min()
    coords /= coords[:, 0].max()

    # 前縁を境に分割
    i_le = np.argmin(coords[:,0])  # 前縁のインデックス
    upper = coords[:i_le+1]
    lower = coords[i_le:]

    # 補間関数
    def resample_curve(curve, n):
        dist = np.cumsum(np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1)))
        dist = np.insert(dist, 0, 0)
        t = dist / dist[-1]
        fx = interpolate.interp1d(t, curve[:, 0])
        fy = interpolate.interp1d(t, curve[:, 1])
        t_new = np.linspace(0, 1, n)
        return np.stack([fx(t_new), fy(t_new)], axis=1)

    # 上面と下面を別々にリサンプリング
    upper_resampled = resample_curve(upper, n_points//2)
    lower_resampled = resample_curve(lower, n_points//2)

    # 結合
    return np.vstack([upper_resampled, lower_resampled[1:]])  # 前縁の重複1点を削除



def process_uiuc_folder(folder_path, n_points=200, save_path="airfoils_resampled.npy"):
    """UIUCのフォルダを処理して固定点数に揃えたデータセットを作成"""
    all_files = glob.glob(os.path.join(folder_path, "*.dat"))
    dataset = []
    names = []

    for i, file in enumerate(all_files):
        try:
            coords = load_airfoil_dat(file)
            coords_resampled = normalize_and_resample(coords, n_points)
            dataset.append(coords_resampled)
            names.append(os.path.basename(file))
        except Exception as e:
            print(f"Skipping {file}: {e}")

    dataset = np.array(dataset)  # shape = (num_files, n_points, 2)
    np.save(save_path, dataset)
    print(f"Saved {dataset.shape} to {save_path}")
    return dataset, names

# 実行例
if __name__ == "__main__":
    folder = "UIUC Database"  # UIUCのdatフォルダ
    dataset, names = process_uiuc_folder(folder, n_points=200)
