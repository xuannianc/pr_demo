import cv2
import numpy as np
import pandas as pd
import os


def split(filepath):
    media_dir = os.path.dirname(filepath)
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    height, width = binary.shape
    # 获取所有竖线
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
    heroded = cv2.erode(binary, k1, iterations=1)
    hdilated = cv2.dilate(heroded, k1, iterations=1)
    # 获取所有横线
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
    veroded = cv2.erode(binary, k2, iterations=1)
    vdilated = cv2.dilate(veroded, k2, iterations=1)
    # 处理横线
    point_array = np.where(hdilated == 255)
    axis0 = pd.Series(point_array[0])
    h_statistics = axis0.value_counts().sort_index()
    ## 合并紧连的横线
    h_pre_index = h_statistics.index[0]
    for index in h_statistics.index:
        if index - h_pre_index <= 3:
            or_lines = hdilated[h_pre_index] | hdilated[index]
            hdilated[h_pre_index] = 0
            hdilated[index] = or_lines
        h_pre_index = index
    ## 延长横线
    point_array2 = np.where(hdilated == 255)
    axis0 = pd.Series(point_array2[0])
    h_statistics = axis0.value_counts().sort_index()
    for index in h_statistics.index:
        col_arr, = np.where(hdilated[index] == 255)
        min_col = np.min(col_arr)
        max_col = np.max(col_arr)
        hdilated[index][min_col - 15: max_col + 15] = 255
    # 处理竖直线
    point_array = np.where(vdilated == 255)
    axis1 = pd.Series(point_array[1])
    v_statistics = axis1.value_counts().sort_index()
    ## 合并紧连的竖线
    v_pre_index = v_statistics.index[0]
    for index in v_statistics.index:
        if index - v_pre_index <= 3:
            or_columns = vdilated[:, v_pre_index] | vdilated[:, index]
            vdilated[:, v_pre_index] = 0
            vdilated[:, index] = or_columns
        v_pre_index = index
    ## 延长线
    point_array2 = np.where(vdilated == 255)
    axis1 = pd.Series(point_array2[1])
    v_statistics = axis1.value_counts().sort_index()
    for index in v_statistics.index:
        row_arr, = np.where(vdilated[:, index] == 255)
        min_row = np.min(row_arr)
        max_row = np.max(row_arr)
        vdilated[min_row - 5:max_row + 5, index] = 255
    # 获取横线竖线交叉点
    cross_point = cv2.bitwise_and(hdilated, vdilated)
    point_array = np.where(cross_point == 255)
    axis0 = pd.Series(point_array[0])
    h_statistics = axis0.value_counts().sort_index()
    all_h = h_statistics.index
    axis1 = pd.Series(point_array[1])
    v_statistics = axis1.value_counts().sort_index()
    all_v = v_statistics.index
    # 分隔图片
    exporter = image[all_h[0] + 1:all_h[5], all_v[0] + 1:all_v[6]]
    cv2.imwrite(os.path.join(media_dir, 'exporter.jpg'), exporter)
    consignee = image[all_h[5] + 1:all_h[6], all_v[0] + 1:all_v[6]]
    cv2.imwrite(os.path.join(media_dir, 'consignee.jpg'), consignee)
    details = image[all_h[0] + 1:all_h[6], all_v[6] + 1:all_v[-1]]
    cv2.imwrite(os.path.join(media_dir, 'details.jpg'), details)
    notify_party = image[all_h[6] + 1:all_h[7], all_v[0] + 1:all_v[6]]
    cv2.imwrite(os.path.join(media_dir, 'notify_party.jpg'), notify_party)
    buyer = image[all_h[6] + 1:all_h[7], all_v[6] + 1:all_v[-1]]
    cv2.imwrite(os.path.join(media_dir, 'buyer.jpg'), buyer)
