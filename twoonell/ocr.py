import cv2
import numpy as np
import pandas as pd
from collections import Counter
import pytesseract

BUFFER_I = 5
BUFFER_II = 10
BUFFER_III = 20
BUFFER_IV = 100


def extract_text(image, line=True):
    if line:
        text = pytesseract.image_to_string(image, config='--psm 4')
    else:
        text = pytesseract.image_to_string(image)
    return text


def get_combined_x(image):
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
    # cv2.imwrite('vdilated2.jpg', vdilated)
    # 处理竖直线
    # 所有白点的 x 坐标
    all_x = np.where(vdilated == 255)[1]
    all_x_s = pd.Series(all_x)
    # 所有白点的 x 坐标统计
    x_statistics = all_x_s.value_counts().sort_index()
    # print(v_statistics)
    all_combined_x = []
    # 合并紧连的竖线
    pre_x = x_statistics.index[0]
    for x in x_statistics.index[1:]:
        if x - pre_x > BUFFER_II:
            all_combined_x.append(pre_x)
            pre_x = x
        else:
            # 合并
            or_columns = vdilated[:, pre_x] | vdilated[:, x]
            # 抹去被合并的其他线条
            vdilated[:, x] = 0
            vdilated[:, pre_x] = or_columns
    all_combined_x.append(pre_x)
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
    pre_y = y_statistics.index[0]
    for y in y_statistics.index[1:]:
        # 横线 y 坐标距离大于 5，不再合并
        if y - pre_y > BUFFER_I:
            all_combined_y.append(pre_y)
            pre_y = y
        else:
            # 合并
            or_lines = hdilated[pre_y] | hdilated[y]
            # 抹去被合并过的横线
            hdilated[y] = 0
            hdilated[pre_y] = or_lines
    all_combined_y.append(pre_y)
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
    print("seqs={}".format(seqs))
    c = Counter(seqs)
    longest_seqs = c.most_common(top)
    print("The {} longest seqs={}".format(top, longest_seqs))
    segment_line_indexes = []
    for seq_start, seq_len in longest_seqs:
        seq_middle = (seq_start + seq_start + seq_len) // 2
        segment_line_indexes.append(seq_middle)
    print("segment_line_indexes={}".format(segment_line_indexes))
    return sorted(segment_line_indexes)


def draw_lines(img, indexes, axis=0):
    H, W = img.shape[:2]
    if axis == 0:
        for idx in indexes:
            cv2.line(img, (idx, 0), (idx, H), (255, 0, 0), 2)
    elif axis == 1:
        for idx in indexes:
            cv2.line(img, (0, idx), (W, idx), (255, 0, 0), 2)
    # cv2.imshow('lines', img)


def extract_invoice_no(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    hist = cv2.reduce(binary, 1, cv2.REDUCE_AVG).reshape(-1)
    th = 3
    H, W = image.shape[:2]
    uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
    lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]

    print('uppers={}'.format(uppers))
    print('lowers={}'.format(lowers))
    # for y in uppers:
    #     cv2.line(image, (0, y), (W, y), (255, 0, 0), 1)
    # for y in lowers:
    #     cv2.line(image, (0, y), (W, y), (0, 255, 0), 1)
    # cv2.imshow('invoice_no_lines', image)
    # cv2.waitKey(0)
    invoice_no_row = binary[uppers[1]:lowers[1], :]
    segment_col_lines = get_segment_lines(invoice_no_row, top=3, axis=0)
    post_invoice_no_img = image[uppers[1]:lowers[1] + BUFFER_II, segment_col_lines[1]:segment_col_lines[2]]
    # 上方插入空白
    post_invoice_no_img = np.insert(post_invoice_no_img, [0] * BUFFER_III, values=255, axis=0)
    # cv2.imshow('post_invoice_no_img', post_invoice_no_img)
    # cv2.waitKey(0)
    invoice_no = extract_text(post_invoice_no_img)
    return invoice_no


def extract_invoice_date(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    segment_col_lines = get_segment_lines(binary, top=3, axis=0)
    # draw_lines(image, segment_col_lines, axis=0)
    # cv2.imshow('pre_invoice_date_lines', image)
    # cv2.waitKey(0)
    post_invoice_date_img = image[:, segment_col_lines[1]:segment_col_lines[2]]
    # cv2.imshow('post_invoice_date_img', post_invoice_date_img)
    # cv2.waitKey(0)
    invoice_date = extract_text(post_invoice_date_img)
    return invoice_date


def extract_order_no(image):
    # cv2.imshow('post_order_no_img', image)
    # cv2.waitKey(0)
    order_no = extract_text(image)
    return order_no


def scan_table(image):
    """
    扫描表中的内容
    :param image: 原图像的表格部分
    :return:
    """
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    all_combined_x = get_combined_x(binary)
    col1 = binary[:, all_combined_x[0] + BUFFER_II:all_combined_x[1] - BUFFER_II]
    hist = cv2.reduce(col1, 1, cv2.REDUCE_AVG).reshape(-1)
    th = BUFFER_I
    uppers = [y for y in range(0, H - 1) if
              hist[y] <= th and hist[y + 1] > th]
    lowers = [y for y in range(0, H - 1) if
              hist[y] > th and hist[y + 1] <= th]
    # print('uppers={}'.format(uppers))
    # print('lowers={}'.format(lowers))
    # 清除所有的条目中的日期
    for i in range(len(uppers) - 1):
        image[lowers[i] + BUFFER_I:uppers[i + 1] - BUFFER_I] = 255
    image[lowers[-1] + BUFFER_I:H - 1] = 255
    # 生成所有的列
    col_names = ['IDENTIFICATION', 'QUANTITY', 'PACKAGING', 'UNITS', 'DESCRIPTION', 'UNIT PRICE', 'TOTAL PRICE']
    df = pd.DataFrame()
    for idx, x in enumerate(all_combined_x[:-1]):
        col = image[:lowers[-1] + BUFFER_II, x + BUFFER_II:all_combined_x[idx + 1] - BUFFER_I]
        # 上方插入空白
        col_with_upper_space = np.insert(col, [0] * BUFFER_II, values=255, axis=0)
        # 右方插入空白
        col_with_right_space = np.insert(col_with_upper_space, [col.shape[1] - 1] * BUFFER_IV, values=255, axis=1)
        # 左方插入空白
        col_with_space = np.insert(col_with_right_space, [0] * BUFFER_IV, values=255, axis=1)
        # cv2.imshow(col_names[idx], col_with_space)
        # cv2.waitKey(0)
        text = extract_text(col_with_space)
        words = [words for words in text.split('\n') if words]
        print(words)
        df[col_names[idx]] = words
    return df.to_dict('records')


def scan_invoice(image):
    """
    扫描左上角的 invoice 信息，包括 invoice no,invoice date,order no
    :param image: 原图像的 invoice 区域
    :return:
    """
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    segment_row_lines = get_segment_lines(binary, top=2, axis=1)
    # draw_lines(image, segment_row_lines, axis=1)
    segment_col_lines = get_segment_lines(binary, top=4, axis=0)
    # draw_lines(image, segment_col_lines, axis=0)
    # cv2.imshow('invoice_lines',image)
    # cv2.waitKey(0)
    pre_invoice_no_img = image[:segment_row_lines[0], segment_col_lines[0]:segment_col_lines[1]]
    # cv2.imshow('invoice_no_pre_img', pre_invoice_no_img)
    # cv2.waitKey(0)
    invoice_no = extract_invoice_no(pre_invoice_no_img)
    print("invoice_no={}".format(invoice_no))
    pre_invoice_date_img = image[segment_row_lines[0]:segment_row_lines[1], segment_col_lines[0]:segment_col_lines[1]]
    # cv2.imshow('invoice_date_pre_img', pre_invoice_date_img)
    # cv2.waitKey(0)
    invoice_date = extract_invoice_date(pre_invoice_date_img)
    print("invoice_date={}".format(invoice_date))
    pre_order_no_img = image[segment_row_lines[1]:H, segment_col_lines[1]:segment_col_lines[-1]]
    # cv2.imshow('order_no_img', pre_order_no_img)
    # cv2.waitKey(0)
    order_no = extract_order_no(pre_order_no_img)
    print("order_no={}".format(order_no))
    return {"invoice_no": invoice_no, "invoice_date": invoice_date, "order_no": order_no}


def scan_address(image):
    H, W = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    hist = cv2.reduce(binary, 1, cv2.REDUCE_AVG).reshape(-1)
    th = 3
    uppers = [y for y in range(H - 1) if hist[y] <= th and hist[y + 1] > th]
    lowers = [y for y in range(H - 1) if hist[y] > th and hist[y + 1] <= th]

    # for y in uppers:
    #     cv2.line(image, (0, y), (W, y), (255, 0, 0), 1)
    # for y in lowers:
    #     cv2.line(image, (0, y), (W, y), (0, 255, 0), 1)
    # cv2.imshow('address_with_lines', image)
    # cv2.waitKey(0)
    text = extract_text(image)
    return text


def scan_without_preprocessing(image):
    text = extract_text(image)
    words = [words for words in text.split('\n') if words]
    return '\n'.join(words)


def scan(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    all_combined_y = get_combined_y(binary)
    scan_result = {}
    # 处理表格
    table = image[all_combined_y[8] + BUFFER_I:all_combined_y[9] - BUFFER_I, :]
    items = scan_table(table)
    print("items={}".format(items))
    scan_result['items'] = items
    # 处理 row2: invoice no,invoice date,order no,invoice address
    row2_img = binary[all_combined_y[2] + BUFFER_I:all_combined_y[3] - BUFFER_I, :]
    # cv2.imshow('invoice_img', row2_img)
    # cv2.waitKey(0)
    row2_combined_x = get_combined_x(row2_img)
    invoice_img = image[all_combined_y[2]:all_combined_y[4] - BUFFER_II, :row2_combined_x[0] - BUFFER_II]
    # cv2.imshow('invoice_img', invoice_img)
    # cv2.waitKey(0)
    invoice = scan_invoice(invoice_img)
    print('invoice={}'.format(invoice))
    scan_result.update(invoice)
    # 处理 invoice address
    invoice_address_img = image[all_combined_y[2] + BUFFER_I:all_combined_y[3] - BUFFER_I,
                          row2_combined_x[0] + BUFFER_II:row2_combined_x[1] - BUFFER_IV]
    # cv2.imshow('invoice_address', invoice_address_img)
    # cv2.waitKey(0)
    invoice_address = scan_address(invoice_address_img)
    print('invoice_address={}'.format(invoice_address))
    scan_result['invoice_address'] = invoice_address
    # 处理 row1,delivery address
    row1_img = binary[all_combined_y[0] + BUFFER_I:all_combined_y[1] - BUFFER_I, :]
    row1_combined_x = get_combined_x(row1_img)
    delivery_address_img = image[all_combined_y[0] + BUFFER_I:all_combined_y[1] - BUFFER_I,
                           row1_combined_x[0] + BUFFER_II:row1_combined_x[1] - BUFFER_IV]
    # cv2.imshow('delivery_address', delivery_address_img)
    # cv2.waitKey(0)
    delivery_address = scan_address(delivery_address_img)
    print('delivery_address={}'.format(delivery_address))
    scan_result['delivery_address'] = delivery_address
    # 处理 row3: settlement,shipment
    row3_img = binary[all_combined_y[5] + BUFFER_I:all_combined_y[6] - BUFFER_I, :]
    row3_combined_x = get_combined_x(row3_img)
    # 处理 settlement
    settlement_img = image[all_combined_y[5] + BUFFER_I:all_combined_y[6] - BUFFER_I,
                     row3_combined_x[0] + BUFFER_I:row3_combined_x[1] - BUFFER_I]
    # cv2.imshow('settlement_img', settlement_img)
    # cv2.waitKey(0)
    settlement = scan_without_preprocessing(settlement_img)
    print('settlement={}'.format(settlement))
    scan_result['settlement'] = settlement
    # 处理 shipment
    shipment_img = image[all_combined_y[5] + BUFFER_I:all_combined_y[6] - BUFFER_I,
                   row3_combined_x[2] + BUFFER_I:row3_combined_x[3] - BUFFER_I]
    # cv2.imshow('shipment_img', shipment_img)
    # cv2.waitKey(0)
    shipment = scan_without_preprocessing(shipment_img)
    print('shipment={}'.format(shipment))
    scan_result['shipment'] = shipment

    # 处理 row5: total cases, total amount due
    row5_img = binary[all_combined_y[11] + BUFFER_I:all_combined_y[12] - BUFFER_I, :]
    row5_combined_x = get_combined_x(row5_img)
    # 处理 total cases
    total_cases_img = image[all_combined_y[11] + BUFFER_I:all_combined_y[12] - BUFFER_I,
                      row5_combined_x[0] + BUFFER_I:row5_combined_x[1] - BUFFER_I]
    # cv2.imshow('total_cases_img', total_cases_img)
    # cv2.waitKey(0)
    total_cases = scan_without_preprocessing(total_cases_img)
    print('total_cases={}'.format(total_cases))
    scan_result['total_cases'] = total_cases
    # 处理 total amount due
    total_amount_due_img = image[all_combined_y[11] + BUFFER_I:all_combined_y[12] - BUFFER_I,
                           row5_combined_x[2] + BUFFER_I:row5_combined_x[3] - BUFFER_I]
    # cv2.imshow('total_amount_due_img', total_amount_due_img)
    # cv2.waitKey(0)
    total_amount_due = scan_without_preprocessing(total_amount_due_img)
    print('total_amount_due'.format(total_amount_due))
    scan_result['total_amount_due'] = total_amount_due
    print(scan_result)
    return scan_result

