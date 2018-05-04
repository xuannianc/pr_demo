import cv2
import numpy as np
import pandas as pd
import os
from collections import Counter
import re

OFFSET = 30
WORDS = Counter(["status",
                 "payment",
                 "no",
                 "prepay",
                 "description",
                 "supplier",
                 "name",
                 "currency",
                 "term",
                 "requested",
                 "date",
                 "bu",
                 "alternate",
                 "payee"])


def preprocessing(filepath):
    media_dir = os.path.dirname(filepath)
    img = cv2.imread(filepath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    height, width = binary.shape
    # 获取所有横线
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 30, 1))
    heroded = cv2.erode(binary, k1, iterations=1)
    hdilated = cv2.dilate(heroded, k1, iterations=1)
    # cv2.imshow('hdilated',hdilated)
    # cv2.waitKey(0)
    # 获取所有竖线
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 30))
    veroded = cv2.erode(binary, k2, iterations=1)
    vdilated = cv2.dilate(veroded, k2, iterations=1)
    # 合并所有紧连的横线
    ## 横线上所有点的坐标
    h_points = np.where(hdilated == 255)
    ## 横线上所有点的 x 轴坐标
    axis0 = pd.Series(h_points[0])
    ## 横线上所有点的 x 轴坐标进行排序
    h_statistics = axis0.value_counts().sort_index()
    print(h_statistics)
    h_pre_index = h_statistics.index[0]
    all_h_index = [h_pre_index]
    for index in h_statistics.index[1:]:
        if index - h_pre_index <= 5:
            ## 在 binary 图上合并
            or_lines = binary[h_pre_index] | binary[index]
            binary[index] = 0
            binary[h_pre_index] = or_lines
            ## 在 hdilated 图上合并，为了获取表格交点
            or_lines = hdilated[h_pre_index] | hdilated[index]
            hdilated[index] = 0
            hdilated[h_pre_index] = or_lines
        else:
            h_pre_index = index
            all_h_index.append(index)
    print(all_h_index)
    ## 竖线上所有点的坐标
    v_points = np.where(vdilated == 255)
    ## 竖线上所有点的 y 轴坐标
    axis1 = pd.Series(v_points[1])
    ## 竖线上所有点的 y 轴坐标进行排序
    v_statistics = axis1.value_counts().sort_index()
    v_pre_index = v_statistics.index[0]
    all_v_index = [v_pre_index]
    for index in v_statistics.index[1:]:
        if index - v_pre_index <= 5:
            ## 在 binary 图上合并
            or_lines = binary[:, v_pre_index] | binary[:, index]
            binary[:, index] = 0
            binary[:, v_pre_index] = or_lines
            ## 在 vdilated 图上合并
            or_lines = vdilated[:, v_pre_index] | vdilated[:, index]
            vdilated[:, index] = 0
            vdilated[:, v_pre_index] = or_lines
        else:
            v_pre_index = index
            all_v_index.append(index)
    # 延长横线
    for index in all_h_index:
        col_arr, = np.where(hdilated[index] == 255)
        min_col = np.min(col_arr)
        max_col = np.max(col_arr)
        hdilated[index, min_col - 5: max_col + 5] = 255
    # 延长竖线
    for index in all_v_index:
        row_arr, = np.where(vdilated[:, index] == 255)
        min_row = np.min(row_arr)
        max_row = np.max(row_arr)
        vdilated[min_row - 5:max_row + 5, index] = 255
    # 完整表格
    or_dilated = cv2.bitwise_or(vdilated, hdilated)
    # 表格的交叉点
    and_dilated = cv2.bitwise_and(vdilated, hdilated)
    # 找出表格左上方顶点和左下方顶点
    cross_points = np.where(and_dilated == 255)
    axis0 = pd.Series(cross_points[0])
    axis1 = pd.Series(cross_points[1])
    h_statistics = axis0.value_counts().sort_index()
    v_statistics = axis1.value_counts().sort_index()
    ## 表格第一条和最后一条横线的下标
    h_first = h_statistics.index[0]
    h_last = h_statistics.index[-1]
    ## 表格第一条和最后一条竖线的下标
    v_first = v_statistics.index[0]
    v_last = v_statistics.index[-1]
    ## 表格第一条横线在所有横线中的 index
    h_first_index = all_h_index.index(h_first)
    ## 表格最后一条横线在所有横线中的 index
    h_last_index = all_h_index.index(h_last)
    ## 表格数据的行数
    rows = h_last_index - 1 - (h_first_index + 1)
    # 处理表格上方内容, part1
    for index in all_h_index[:h_first_index]:
        ## 删除所有横线
        binary[index] = 0
    ## 截取文件,part1
    part1 = binary[:h_first - OFFSET, :]
    part1 = cv2.bitwise_not(part1)
    blurred_part1 = cv2.GaussianBlur(part1, (3, 3), 0)
    cv2.imwrite(os.path.join(media_dir, 'blurred_part1.jpg'), blurred_part1)
    request_part1 = binary[:h_first - OFFSET, :]
    request_part1 = cv2.bitwise_not(request_part1)
    cv2.imwrite(os.path.join(media_dir, 'request_part1.jpg'), request_part1)
    # 处理表格下方内容, part3
    for index in all_h_index[h_last_index + 1:]:
        ## 删除所有横线
        binary[index] = 0
    ## 截取文件,part3
    part3 = binary[h_last + OFFSET:all_h_index[-1] + OFFSET, :]
    part3 = cv2.bitwise_not(part3)
    blurred_part3 = cv2.GaussianBlur(part3, (3, 3), 0)
    cv2.imwrite(os.path.join(media_dir, 'blurred_part3.jpg'), blurred_part3)
    request_part3 = binary[h_last + OFFSET:all_h_index[-1] + OFFSET, :]
    request_part3 = cv2.bitwise_not(request_part3)
    cv2.imwrite(os.path.join(media_dir, 'request_part3.jpg'), request_part3)
    request_part2 = gray[h_first - 1:h_last + 1, :]
    cv2.imwrite(os.path.join(media_dir, 'request_part2.jpg'), request_part2)
    # 处理表格
    ## 处理横线
    for index in all_h_index[h_last_index - 1:h_first_index:-1]:
        ### 去掉横线
        binary[index] = 0
        ### 下方填充空白行
        binary = np.insert(binary, [index + 1] * OFFSET, 0, axis=0)
        ### 上方填充空白行
        binary = np.insert(binary, [index - 1] * OFFSET, 0, axis=0)
    ## 处理竖线
    for index in all_v_index[::-1]:
        ### 去掉竖线
        binary[:, index] = 0
        ### 右方填充空白列
        binary = np.insert(binary, [index + 1] * OFFSET, 0, axis=1)
        ### 左方填充空白列
        binary = np.insert(binary, [index - 1] * OFFSET, 0, axis=1)

    col01 = binary[
            all_h_index[h_first_index + 1] + OFFSET + 1:all_h_index[h_last_index - 1] + OFFSET * (2 * rows + 1) - 1,
            all_v_index[0] + OFFSET:all_v_index[2] + OFFSET * (2 * 2 + 1)]
    col2 = binary[
           all_h_index[h_first_index + 1] + OFFSET + 1:all_h_index[h_last_index - 1] + OFFSET * (2 * rows + 1) - 1,
           all_v_index[2] + OFFSET * (2 * 2 + 1):all_v_index[3] + OFFSET * (3 * 2 + 1)]
    col3456 = binary[
              all_h_index[h_first_index + 1] + OFFSET + 1:all_h_index[h_last_index - 1] + OFFSET * (2 * rows + 1) - 1,
              all_v_index[3] + OFFSET * (3 * 2 + 1):all_v_index[7] + OFFSET * (7 * 2 + 1)]
    col01 = cv2.bitwise_not(col01)
    blurred_col01 = cv2.GaussianBlur(col01, (3, 3), 0)
    cv2.imwrite(os.path.join(media_dir, 'blurred_col01.jpg'), blurred_col01)
    cv2.imwrite(os.path.join(media_dir, 'col01.jpg'), col01)
    col2 = cv2.bitwise_not(col2)
    blurred_col2 = cv2.GaussianBlur(col2, (3, 3), 0)
    cv2.imwrite(os.path.join(media_dir, 'blurred_col2.jpg'), blurred_col2)
    cv2.imwrite(os.path.join(media_dir, 'col2.jpg'), col2)
    col3456 = cv2.bitwise_not(col3456)
    blurred_col3456 = cv2.GaussianBlur(col3456, (3, 3), 0)
    cv2.imwrite(os.path.join(media_dir, 'blurred_col3456.jpg'), blurred_col3456)
    cv2.imwrite(os.path.join(media_dir, 'col3456.jpg'), col3456)
    return h_last_index - h_first_index - 2


def preprocessing2(filepath):
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
    # 合并所有紧连的横线
    ## 横线上所有点的坐标
    h_points = np.where(hdilated == 255)
    ## 横线上所有点的 x 轴坐标
    axis0 = pd.Series(h_points[0])
    ## 横线上所有点的 x 轴坐标进行排序
    h_statistics = axis0.value_counts().sort_index()
    print(h_statistics)
    h_pre_index = h_statistics.index[0]
    all_h_index = [h_pre_index]
    for index in h_statistics.index[1:]:
        if index - h_pre_index <= 5:
            ## 在 gray 图上合并
            or_lines = gray[h_pre_index] & gray[index]
            gray[index] = 255
            gray[h_pre_index] = or_lines
            ## 在 binary 图上合并
            or_lines = binary[h_pre_index] & binary[index]
            binary[index] = 0
            binary[h_pre_index] = or_lines
            ## 在 hdilated 图上合并，为了获取表格交点
            or_lines = hdilated[h_pre_index] | hdilated[index]
            hdilated[index] = 0
            hdilated[h_pre_index] = or_lines
        else:
            h_pre_index = index
            all_h_index.append(index)
    print(all_h_index)
    ## 竖线上所有点的坐标
    v_points = np.where(vdilated == 255)
    ## 竖线上所有点的 y 轴坐标
    axis1 = pd.Series(v_points[1])
    ## 竖线上所有点的 y 轴坐标进行排序
    v_statistics = axis1.value_counts().sort_index()
    v_pre_index = v_statistics.index[0]
    all_v_index = [v_pre_index]
    for index in v_statistics.index[1:]:
        if index - v_pre_index <= 5:
            ## 在 gray 图上合并
            or_lines = gray[:, v_pre_index] & gray[:, index]
            gray[:, index] = 255
            gray[:, v_pre_index] = or_lines
            ## 在 binary 图上合并
            or_lines = binary[:, v_pre_index] & binary[:, index]
            binary[:, index] = 0
            binary[:, v_pre_index] = or_lines
            ## 在 vdilated 图上合并
            or_lines = vdilated[:, v_pre_index] | vdilated[:, index]
            vdilated[:, index] = 0
            vdilated[:, v_pre_index] = or_lines
        else:
            v_pre_index = index
            all_v_index.append(index)
    # 延长横线
    for index in all_h_index:
        col_arr, = np.where(hdilated[index] == 255)
        min_col = np.min(col_arr)
        max_col = np.max(col_arr)
        hdilated[index, min_col - 5: max_col + 5] = 255
    # 延长竖线
    for index in all_v_index:
        row_arr, = np.where(vdilated[:, index] == 255)
        min_row = np.min(row_arr)
        max_row = np.max(row_arr)
        vdilated[min_row - 5:max_row + 5, index] = 255
    # 完整表格
    or_dilated = cv2.bitwise_or(vdilated, hdilated)
    # 表格的交叉点
    and_dilated = cv2.bitwise_and(vdilated, hdilated)
    # 找出表格左上方顶点和左下方顶点
    cross_points = np.where(and_dilated == 255)
    axis0 = pd.Series(cross_points[0])
    axis1 = pd.Series(cross_points[1])
    h_statistics = axis0.value_counts().sort_index()
    v_statistics = axis1.value_counts().sort_index()
    ## 表格第一条和最后一条横线的下标
    h_first = h_statistics.index[0]
    h_last = h_statistics.index[-1]
    ## 表格第一条和最后一条竖线的下标
    v_first = v_statistics.index[0]
    v_last = v_statistics.index[-1]
    ## 表格第一条横线在所有横线中的 index
    h_first_index = all_h_index.index(h_first)
    ## 表格最后一条横线在所有横线中的 index
    h_last_index = all_h_index.index(h_last)
    ## 表格数据的行数
    rows = h_last_index - 1 - (h_first_index + 1)
    # 处理表格上方内容, part1
    for index in all_h_index[:h_first_index]:
        ## 删除所有横线
        gray[index] = 255
    ## 截取文件,part1
    request_part1 = gray[:h_first - OFFSET, :]
    cv2.imwrite(os.path.join(media_dir, 'request_part1.jpg'), request_part1)
    # 处理表格下方内容, part3
    for index in all_h_index[h_last_index + 1:]:
        ## 删除所有横线
        gray[index] = 255
    ## 截取文件,part3
    request_part3 = gray[h_last + OFFSET:all_h_index[-1] + OFFSET, :]
    cv2.imwrite(os.path.join(media_dir, 'request_part3.jpg'), request_part3)
    request_part2 = gray[h_first - 1:h_last + 1, :]
    cv2.imwrite(os.path.join(media_dir, 'request_part2.jpg'), request_part2)
    # 处理表格
    ## 处理横线
    for index in all_h_index[h_last_index - 1:h_first_index:-1]:
        ### 去掉横线
        gray[index] = 255
        ### 下方填充空白行
        gray = np.insert(gray, [index + 1] * OFFSET, 255, axis=0)
        ### 上方填充空白行
        gray = np.insert(gray, [index - 1] * OFFSET, 255, axis=0)
    ## 处理竖线
    for index in all_v_index[::-1]:
        ### 去掉竖线
        gray[:, index] = 255
        ### 右方填充空白列
        gray = np.insert(gray, [index + 1] * OFFSET, 255, axis=1)
        ### 左方填充空白列
        gray = np.insert(gray, [index - 1] * OFFSET, 255, axis=1)

    ## 为表格的每个列创建图片文件
    for i in range(7):
        request_col = gray[all_h_index[
                               h_first_index + 1] + OFFSET + 1: all_h_index[h_last_index - 1] + OFFSET * (
                2 * rows + 1) - 1,
                      all_v_index[i] + OFFSET * (2 * i + 1): all_v_index[i + 1] + OFFSET * (2 * (i + 1) + 1) - 1]
        cv2.imwrite(os.path.join(media_dir, 'request_col' + str(i) + '.jpg'), request_col)
    return h_last_index - h_first_index - 2


def P(word, N=sum(WORDS.values())):
    "Probability of `word`."
    return WORDS[word] / N


def correction(word):
    "Most probable spelling correction for word."
    # 只处理长度大于 2 的字符串
    if len(word) > 2:
        return max(candidates(word), key=P)
    else:
        return word


def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters = 'abcdefghijklmnopqrstuvwxyz'
    # 把单词拆分成两段
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    # print('splits:', splits)
    # print(''.join(['-'] * 50))
    # 删除一个字母
    deletes = [L + R[1:] for L, R in splits if R]
    # print('deletes', deletes)
    # print(''.join(['-'] * 50))
    # 相连两个字母交换位置
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    # print('transposes', transposes)
    # print(''.join(['-'] * 50))
    # 把每一个字母分别替换成其他 25 个字母
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    # print('replaces', replaces)
    # print(''.join(['-'] * 50))
    inserts = [L + c + R for L, R in splits for c in letters]
    # 在每一个位置分别插入 26 个字母中的一个
    # print('inserts', inserts)
    # print(''.join(['-'] * 50))
    return set(deletes + transposes + replaces + inserts)


def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
