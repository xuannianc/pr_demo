import cv2
import numpy as np
import pandas as pd
import pytesseract
from collections import Counter
import os

BUFFER_I = 5
BUFFER_II = 10
BUFFER_III = 20
BUFFER_IV = 100

ITEM_NAMES = [
    ['收发货人', '进口口岸', '进口日期', '申报日期'],
    ['消费使用单位', '运输方式', '运输工具名称', '提运单号'],
    ['申报单位', '监管方式', '征免性质', '备案号'],
    ['贸易国(地区)', '启运国(地区)', '装货港', '境内目的地'],
    ['许可证号', '成交方式', '运费', '保费', '杂费'],
    ['合同协议号', '件数', '包装种类', '毛重(千克)', '净重(千克)'],
    ['集装箱号', '随附单证'],
    ['标记唛码及备注'],
    ['项号', '商品编号', '商品名称、规格型号', '数量及单位', '原产国（地区）', '单价', '总价', '币制', '征免'],
    ['特殊关系确认', '价格影响确认', '与货物有关的特许权使用费支付确认']
]


def extract_text(image, lang='chi_sim', psm=6, oem=1, is_digit=False):
    if is_digit:
        text = pytesseract.image_to_string(image,
                                           config='--psm {} --oem {} -c tessedit_char_whitelist=0123456789'.format(psm,
                                                                                                                   oem))
    else:
        text = pytesseract.image_to_string(image, lang=lang, config='--psm {} --oem {}'.format(psm, oem))
    return text


def clear_horizontal_lines(image, binary):
    H, W = image.shape[:2]
    # 获取所有横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // BUFFER_II, 1))
    heroded = cv2.erode(binary, kernel, iterations=1)
    hdilated = cv2.dilate(heroded, kernel, iterations=1)
    image[hdilated == 255] = 255
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    return image


def get_combined_x(image):
    """
    获取合并后所有竖线的 x 坐标
    :param image:要处理的已经二值化取反后的图像
    :return:
    """
    H, W = image.shape
    # 获取所有竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H - BUFFER_II))
    veroded = cv2.erode(image, kernel, iterations=1)
    vdilated = cv2.dilate(veroded, kernel, iterations=1)
    # 处理竖直线
    # 所有白点的 x 坐标
    all_x = np.where(vdilated == 255)[1]
    all_x_s = pd.Series(all_x)
    # 所有白点的 x 坐标统计
    x_statistics = all_x_s.value_counts().sort_index()
    # print(v_statistics)
    indexes = x_statistics.index.tolist()
    all_combined_x = []
    # 合并紧连的竖线
    seq_begin = indexes[0]
    seq_end = indexes[0]
    for idx, i in enumerate(indexes[:-1]):
        # 不相连 index
        if indexes[idx + 1] != i + 1:
            seq_end = i
            seq_middle = (seq_begin + seq_end) // 2
            # all_combined_x.append((seq_begin, seq_middle, seq_end))
            all_combined_x.append(seq_middle)
            seq_begin = indexes[idx + 1]
    seq_end = indexes[-1]
    seq_middle = (seq_begin + seq_end) // 2
    all_combined_x.append(seq_middle)
    # print("all_combined_x={}".format(all_combined_x))
    return all_combined_x


def get_combined_x_v2(image):
    """
    获取合并后所有竖线的 x 坐标
    :param image:要处理的已经二值化取反后的图像
    :return:
    """
    H, W = image.shape
    # 获取所有竖线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, H - BUFFER_III))
    veroded = cv2.erode(image, kernel, iterations=1)
    vdilated = cv2.dilate(veroded, kernel, iterations=1)
    # 处理竖直线
    # 所有白点的 x 坐标
    all_x = np.where(vdilated == 255)[1]
    all_x_s = pd.Series(all_x)
    # 所有白点的 x 坐标统计
    x_statistics = all_x_s.value_counts().sort_index()
    # print(v_statistics)
    indexes = x_statistics.index.tolist()
    all_combined_x = []
    # 合并紧连的竖线
    seq_begin = indexes[0]
    seq_end = indexes[0]
    for idx, i in enumerate(indexes[:-1]):
        # 不相连 index
        if indexes[idx + 1] - i > BUFFER_III:
            seq_end = i
            seq_middle = (seq_begin + seq_end) // 2
            all_combined_x.append(seq_middle)
            seq_begin = indexes[idx + 1]
    seq_end = indexes[-1]
    seq_middle = (seq_begin + seq_end) // 2
    all_combined_x.append(seq_middle)
    print('all_combined_x={}'.format(all_combined_x))
    # cv2.imshow('vdilated', vdilated)
    # cv2.waitKey(0)
    return all_combined_x


def get_combined_y(image):
    """
    获取合并后所有横线的 y 坐标
    :param image: 要处理的已经二值化取反后的图像
    :return:
    """
    H, W = image.shape
    # 获取所有横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // BUFFER_II, 1))
    heroded = cv2.erode(image, kernel, iterations=1)
    hdilated = cv2.dilate(heroded, kernel, iterations=1)
    # 处理横线
    # 所有白点的 y 坐标
    all_y = np.where(hdilated == 255)[0]
    all_y_s = pd.Series(all_y)
    # 所有白点的 y 坐标的统计
    y_statistics = all_y_s.value_counts().sort_index()
    # print(y_statistics)
    all_combined_y = []
    # 合并紧连的横线
    indexes = y_statistics.index.tolist()
    print(indexes)
    seq_begin = indexes[0]
    seq_end = indexes[0]
    for idx, i in enumerate(indexes[:-1]):
        # 不相连 index
        if indexes[idx + 1] != i + 1:
            seq_end = i
            seq_middle = (seq_begin + seq_end) // 2
            all_combined_y.append(seq_middle)
            # all_combined_y.append((seq_begin, seq_middle, seq_end))
            seq_begin = indexes[idx + 1]
    seq_end = indexes[-1]
    seq_middle = (seq_begin + seq_end) // 2
    all_combined_y.append(seq_middle)
    print(all_combined_y)
    print(len(all_combined_y))
    # 合并后所有横线的 y 坐标
    return all_combined_y


def get_combined_y_v2(image):
    """
    获取合并后所有横线的 y 坐标
    :param image: 要处理的已经二值化取反后的图像
    :return:
    """
    H, W = image.shape
    # 获取所有横线
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (W // 50, 1))
    heroded = cv2.erode(image, kernel, iterations=1)
    hdilated = cv2.dilate(heroded, kernel, iterations=1)
    # cv2.imshow('hdilated', hdilated)
    # cv2.waitKey(0)
    # 处理横线
    # 所有白点的 y 坐标
    all_y = np.where(hdilated == 255)[0]
    all_y_s = pd.Series(all_y)
    # 所有白点的 y 坐标的统计
    y_statistics = all_y_s.value_counts().sort_index()
    # print(y_statistics)
    all_combined_y = []
    # 合并紧连的横线
    indexes = y_statistics.index.tolist()
    print(indexes)
    seq_begin = indexes[0]
    seq_end = indexes[0]
    for idx, i in enumerate(indexes[:-1]):
        # 不相连 index
        if indexes[idx + 1] - i > BUFFER_III:
            seq_end = i
            seq_middle = (seq_begin + seq_end) // 2
            all_combined_y.append(seq_middle)
            seq_begin = indexes[idx + 1]
    seq_end = indexes[-1]
    seq_middle = (seq_begin + seq_end) // 2
    all_combined_y.append(seq_middle)
    print(all_combined_y)
    print(len(all_combined_y))
    # 合并后所有横线的 y 坐标
    return all_combined_y


def get_segment_lines(img, top=3, axis=0):
    """
    获取图像空白区域的分割线
    :param img: 被操作的图像, numpy 二维数组
    :param top: 选择连续空白行或者列最多的 n 个空白区域
    :param axis: 1 表示水平分割线, 0 表示垂直分割线
    :return: 空白区域中心线的 index
    """
    df = pd.DataFrame(img)
    uniques = df.nunique(axis=axis)
    indexes = uniques[uniques == 1].index.tolist()
    seq_len = 1
    seq_start = 0
    seqs = {}
    for idx, i in enumerate(indexes[:-2]):
        # 相连 index
        if indexes[idx + 1] == i + 1:
            seq_len += 1
        # 不相连 index
        else:
            seqs[seq_start] = seq_len
            seq_start = indexes[idx + 1]
            seq_len = 1
    # 最后一个 seq
    if seq_len != 1:
        seqs[seq_start] = seq_len
    # print("seqs={}".format(seqs))
    c = Counter(seqs)
    longest_seqs = c.most_common(top)
    # print("The {} longest seqs={}".format(top, longest_seqs))
    segment_line_indexes = []
    for seq_start, seq_len in longest_seqs:
        seq_middle = (seq_start + seq_start + seq_len) // 2
        segment_line_indexes.append(seq_middle)
    # print("segment_line_indexes={}".format(segment_line_indexes))
    return sorted(segment_line_indexes)


def get_concatenated_image(col_value_images):
    sumed_height = sum(col_value_image.shape[0] for col_value_image in col_value_images)
    max_width = max(col_value_image.shape[1] for col_value_image in col_value_images)
    concatenated_image = np.full((sumed_height, max_width, 3), 255, dtype=np.uint8)
    y = 0
    for col_value_image in col_value_images:
        h, w, d = col_value_image.shape
        concatenated_image[y:y + h, 0:w] = col_value_image
        y += h
    return concatenated_image


def draw_lines(image, indexes, axis=0):
    H, W = image.shape[:2]
    if axis == 0:
        for idx in indexes:
            cv2.line(image, (idx, 0), (idx, H), (255, 0, 0), 2)
    elif axis == 1:
        for idx in indexes:
            cv2.line(image, (0, idx), (W, idx), (255, 0, 0), 2)
    cv2.imshow('lines', image)
    cv2.waitKey(0)


def scan_row_integrally(image, row_idx):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    col_names = ITEM_NAMES[row_idx]
    all_combined_x = get_combined_x(binary)
    for x in all_combined_x:
        image[:, x - BUFFER_II: x + BUFFER_II] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    hist = cv2.reduce(binary, 1, cv2.REDUCE_AVG).reshape(-1)
    th = BUFFER_II
    H, W = image.shape[:2]
    uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
    lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]
    # print('uppers{}={}'.format(row_idx, uppers))
    # print('lowers{}={}'.format(row_idx, lowers))
    value_image = image[uppers[1] - BUFFER_II:, :]
    # draw_lines(image, uppers, axis=1)
    # draw_lines(image, lowers, axis=1)
    col_images = []
    for idx, x in enumerate(all_combined_x[:-1]):
        col_images.append(value_image[:, x:all_combined_x[idx + 1]])
    sumed_height = sum(col_image.shape[0] for col_image in col_images)
    max_width = max(col_image.shape[1] for col_image in col_images)
    concatenated_image = np.full((sumed_height, max_width, 3), 255, dtype=np.uint8)
    y = 0
    for col_image in col_images:
        h, w, d = col_image.shape
        concatenated_image[y:y + h, 0:w] = col_image
        y += h
    # cv2.imshow('concatenated_image', concatenated_image)
    # cv2.waitKey(0)
    text = extract_text(concatenated_image)
    col_values = [col_value for col_value in text.split('\n') if col_value]
    scan_row_result = dict(zip(col_names, col_values))
    print('scan_row_result{}={}'.format(row_idx, scan_row_result))
    return scan_row_result


def scan_row_divisionally(image, row_idx):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    all_combined_x = get_combined_x(binary)
    for x in all_combined_x:
        image[:, x - BUFFER_II: x + BUFFER_II] = 255
    # draw_lines(image, all_combined_x, axis=0)
    scan_row_result = {}
    col_value_images = []
    col_names = []
    for idx, x in enumerate(all_combined_x[:-1]):
        col_image = image[:, x + BUFFER_II:all_combined_x[idx + 1] - BUFFER_II]
        col_name = ITEM_NAMES[row_idx][idx]
        col_gray = cv2.cvtColor(col_image, cv2.COLOR_BGR2GRAY)
        col_inv = cv2.threshold(col_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        hist = cv2.reduce(col_inv, 1, cv2.REDUCE_AVG).reshape(-1)
        th = BUFFER_I
        H, W = col_image.shape[:2]
        uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
        lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]
        print('uppers{}{}={}'.format(row_idx, idx, uppers))
        print('lowers{}{}={}'.format(row_idx, idx, lowers))
        # draw_lines(col_image, uppers, axis=1)
        # draw_lines(col_image, lowers, axis=1)
        if len(uppers) == 2 and len(lowers) == 2:
            col_value_image = col_image[uppers[1] - BUFFER_II:, :]
            col_value_images.append(col_value_image)
            col_names.append(col_name)
            # cv2.imshow('col_value_image-{}'.format(idx), col_value_image)
            # cv2.waitKey(0)
        else:
            scan_row_result[col_name] = ''
            # col_content = extract_text(col_image)
            # cleaned_col_content = [line for line in col_content.split('\n') if line]
            # if len(cleaned_col_content) > 1:
            #     scan_row_result[col_name] = '\n'.join(cleaned_col_content[1:])
            # elif len(cleaned_col_content) == 1:
            #     scan_row_result[col_name] = ''
            # else:
            #     print('Error: col{}-{} is empty'.format(row_idx, idx))
    if len(col_value_images) > 0:
        concatenated_image = get_concatenated_image(col_value_images)
        # cv2.imshow('concatenated_image-{}'.format(row_idx), concatenated_image)
        # cv2.waitKey(0)
        text = extract_text(concatenated_image)
        col_values = [col_value for col_value in text.split('\n') if col_value]
        scan_row_result.update(dict(zip(col_names, col_values)))
    print('scan_row_result{}={}'.format(row_idx, scan_row_result))
    return scan_row_result


def scan_table_item(image, item_idx, all_segment_x):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    col_names = ITEM_NAMES[8]
    if not all_segment_x:
        all_segment_x = get_segment_lines(binary, top=8, axis=0)
        # draw_lines(image, all_segment_x, axis=0)
    scan_item_result = {}
    concatenate_col_images = []
    concatenate_col_names = []
    for idx, x in enumerate(all_segment_x):
        if idx == len(all_segment_x) - 1:
            col_image = image[:, x:]
            # cv2.imshow('col{}'.format(idx), col)
            # cv2.waitKey(0)
            col_value = extract_text(col_image)
            scan_item_result[col_names[idx + 1]] = col_value
            if len(concatenate_col_images) > 0:
                concatenated_image = get_concatenated_image(concatenate_col_images)
                # cv2.imshow('concatenated_image-{}'.format(row_idx), concatenated_image)
                # cv2.waitKey(0)
                col_values = extract_text(concatenated_image)
                cleaned_col_values = [col_value for col_value in col_values.split('\n') if col_value]
                scan_item_result.update(dict(zip(concatenate_col_names, cleaned_col_values)))
            print('scan_item_result{}={}'.format(item_idx, scan_item_result))
            return True, scan_item_result, all_segment_x
        elif idx == 0:
            col_image = image[:, x:all_segment_x[idx + 1]]
            # cv2.imshow('col{}'.format(idx), col)
            # cv2.waitKey(0)
            try:
                text = extract_text(col_image, psm=10, oem=0, is_digit=True)
                if text and text == str(item_idx + 1):
                    scan_item_result[col_names[idx]] = text
                else:
                    return False, None, None
            except Exception as e:
                print(e)
                return False, None, None
        elif idx == 3:
            col_image = image[:, x:all_segment_x[idx + 1]]
            # cv2.imshow('col{}'.format(idx), col)
            # cv2.waitKey(0)
            col_segment_lines = get_segment_lines(binary[:, x:all_segment_x[idx + 1]], top=4)
            # draw_lines(col, col_segment_lines)
            col3_image = col_image[:, col_segment_lines[0]: col_segment_lines[-2]]
            col3_value = extract_text(col3_image)
            scan_item_result[col_names[idx]] = col3_value
            col4_image = col_image[:, col_segment_lines[-2]: col_segment_lines[-1]]
            col4_value = extract_text(col4_image)
            scan_item_result[col_names[idx + 1]] = col4_value
        else:
            col_image = image[:, x:all_segment_x[idx + 1]]
            # cv2.imshow('col{}'.format(idx), col)
            # cv2.waitKey(0)
            if idx in [1, 4, 5]:
                concatenate_col_images.append(col_image)
                if idx == 1:
                    concatenate_col_names.append(col_names[idx])
                else:
                    concatenate_col_names.append(col_names[idx + 1])
            elif idx == 2:
                col_value = extract_text(col_image)
                scan_item_result[col_names[idx]] = col_value
            else:
                col_value = extract_text(col_image)
                scan_item_result[col_names[idx + 1]] = col_value


def scan_table(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    all_combined_y = get_combined_y_v2(binary)
    # draw_lines(image, all_combined_y, axis=1)
    all_combined_x = get_combined_x_v2(binary)
    # draw_lines(image, all_combined_x, axis=0)
    table_content = []
    all_segment_x = None
    do_continue = True
    for idx, y in enumerate(all_combined_y):
        if idx == 0:
            item_image = image[:all_combined_y[idx] - BUFFER_II,
                         all_combined_x[0] + BUFFER_II:all_combined_x[1] - BUFFER_II]
        elif idx == len(all_combined_y) - 1:
            item_image = image[all_combined_y[idx] + BUFFER_II:,
                         all_combined_x[0] + BUFFER_II:all_combined_x[1] - BUFFER_II]
        else:
            item_image = image[all_combined_y[idx - 1] + BUFFER_II:all_combined_y[idx] - BUFFER_II,
                         all_combined_x[0] + BUFFER_II:all_combined_x[1] - BUFFER_II]
        # cv2.imshow('item_image{}'.format(idx), item_image)
        # cv2.waitKey(0)
        do_continue, item_content, all_segment_x = scan_table_item(item_image, idx, all_segment_x)
        if do_continue:
            table_content.append(item_content)
        else:
            break

            # kernel = np.ones((1, 100), np.uint8)  # note this is a horizontal kernel
            # hdilated = cv2.dilate(binary, kernel, iterations=1)
            # heroded = cv2.erode(hdilated, kernel, iterations=1)
            # cv2.imshow('heroded',heroded)
            # cv2.waitKey(0)
            # cross_point = cv2.bitwise_and(vdilated,heroded)
            # cv2.imshow('cross_point',cross_point)
            # cv2.waitKey(0)
    return table_content


def scan(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    scan_result = {}
    all_combined_y = get_combined_y(binary)
    image = clear_horizontal_lines(image, binary)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    # draw_lines(image, all_combined_y, axis=1)
    # 处理表格上面的所有行
    for row_idx in range(8):
        row_image = image[all_combined_y[row_idx] + BUFFER_II:all_combined_y[row_idx + 1] - BUFFER_I]
        if row_idx in [0, 3, 5]:
            scan_row_result = scan_row_integrally(row_image, row_idx)
        elif row_idx in [1, 2, 4, 6]:
            scan_row_result = scan_row_divisionally(row_image, row_idx)
        else:
            row_content = extract_text(row_image)
            cleaned_row_content = [words for words in row_content.split('\n') if words]
            if len(cleaned_row_content) > 1:
                row_value = cleaned_row_content[1:]
            else:
                row_value = ''
            scan_row_result = {ITEM_NAMES[row_idx][0]: row_value}
        scan_result['row' + str(row_idx)] = scan_row_result
    # 处理表格
    table_content = scan_table(image[all_combined_y[9] + BUFFER_II: all_combined_y[10] - BUFFER_III])
    scan_result['items'] = table_content
    print('scan_result={}'.format(scan_result))
    return scan_result

# image = cv2.imread('aligned_sh-2.jpg')
# scan(image)
