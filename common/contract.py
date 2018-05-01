import cv2
import numpy as np
import pandas as pd
import os


def split(filepath):
    media_dir = os.path.dirname(filepath)
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    height, width = binary.shape
    # 获取所有横线
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
    heroded = cv2.erode(binary, k1, iterations=1)
    hdilated = cv2.dilate(heroded, k1, iterations=1)
    # 获取所有竖线
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
    veroded = cv2.erode(binary, k2, iterations=1)
    vdilated = cv2.dilate(veroded, k2, iterations=1)
    # 竖线和横线交叉点
    cross_point = cv2.bitwise_and(vdilated, hdilated)
    or_dilated = cv2.bitwise_or(vdilated, hdilated)
    # 去掉竖线和横线
    result = cv2.bitwise_xor(binary, or_dilated)
    # 颜色取反
    result_inv = cv2.bitwise_not(result)
    # 滤波
    # blur = cv2.GaussianBlur(result_inv, (3, 3), 0)
    # 先找第 0 条竖线和第 0 条横线的交点的坐标
    point_array = np.where(cross_point == 255)
    axis0 = pd.Series(point_array[0])
    h_statistics = axis0.value_counts().sort_index()
    all_h = []
    h_pre_index = h_statistics.index[0]
    all_h.append(h_pre_index)
    for index in h_statistics.index:
        if index - h_pre_index > 5:
            all_h.append(index)
            h_pre_index = index
    all_v = []
    axis1 = pd.Series(point_array[1])
    v_statistics = axis1.value_counts().sort_index()
    print(v_statistics)
    v_pre_index = v_statistics.index[0]
    all_v.append(v_pre_index)
    for index in v_statistics.index:
        if index - v_pre_index > 5:
            all_v.append(index)
            v_pre_index = index
    # 生成 part1,part2,part3
    cv2.imwrite(os.path.join(media_dir, 'contract_part1.jpg'), gray[:all_h[0] - 1, :])
    cv2.imwrite(os.path.join(media_dir, 'contract_part2.jpg'), gray[all_h[0] + 1:all_h[-1] - 1, :])
    cv2.imwrite(os.path.join(media_dir, 'contract_part3.jpg'), gray[all_h[-1] + 1:, :])
    # 生成 item1234,item5678,item9101112
    # 获取第二行
    item1234 = gray[all_h[1]:all_h[2], all_v[0]:all_v[-1]]
    # 第一个竖线前后各插入 10 个空白列
    for i in range(len(all_v) - 2, 0, -1):
        with_right_space = np.insert(item1234, [all_v[i] - all_v[0] + 5] * 30, values=255, axis=1)
        with_both_space = np.insert(with_right_space, [all_v[i] - all_v[0] - 5] * 30, values=255, axis=1)
        item1234 = with_both_space
    cv2.imwrite(os.path.join(media_dir, 'item1234.jpg'), item1234)
    item5678 = gray[all_h[3]:all_h[-1], all_v[0]:all_v[2]]
    cv2.imwrite(os.path.join(media_dir, 'item5678.jpg'), item5678)
    item9101112 = gray[all_h[3]:all_h[-1], all_v[2]:all_v[-1]]
    cv2.imwrite(os.path.join(media_dir, 'item9101112.jpg'), item9101112)
