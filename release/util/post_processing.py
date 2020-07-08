import os
import re
import cv2
import numpy as np


# date_reg = re.compile(r'(\d{4}([-/])\d{2}\2\d{2})')
# date_reg = re.compile(r'(\d{4}([-/]?)\d{2}\2\d{2})')
date_reg = re.compile(r"(\d{2,4}-\d{1,2}-\d{1,2})")
id_reg = re.compile(r"(\d{12,22})")

def text_only_information_handle(texts, boxes):
    if len(texts) < 20:  # 不少于20个信息
        return False, None

    if len(texts) != len(boxes):
        return False, None

    lengths = []
    heights = []
    for x0, y0, x1, y1 in boxes:
        lengths.append(x1-x0)
        heights.append(y1-y0)

    avg_l = sum(lengths) // len(lengths)
    avg_h = sum(heights) // len(heights)

    driver_id = ''
    name = ''
    gender = ''
    nation = '中国'
    address = ''
    brith = ''
    first = ''
    class_ = ''
    start = ''
    end = ''
    logo = ''

    # data = [x.split(' ')[0] for x in texts]
    infomation = ','.join(texts)


    # found date
    for i, text in enumerate(texts):
        date = date_reg.findall(text)
        if date:
            if not brith:
                brith = date[0]
                brith = date_validate(brith)
                index_brith = i
                box_length_brith = len(text)
            elif not first:
                first = date[0]
                first = date_validate(first)
                index_first = i
                box_length_first = len(text)
            elif not start:
                start = date[0]
                start = date_validate(start)
                index_start = i
                box_length_start = len(text)
            elif not end or end in ['10年', '6年', '长期']:
                end = date[0]
                end = date_validate(end)
                index_end = i
                box_length_end = len(text)
        if i>15 and not end:
            if '10年' in text or '70年' in text:
                end = '10年'
                index_end = i
                box_length_end = len(text)
            elif '6年' in text:
                end = '6年'
                index_end = i
                box_length_end = len(text)
            elif '长期' in text:
                end = '长期'
                index_end = i
                box_length_end = len(text)
            elif '年' in text:
                if len(text) == 2:
                    end = '6年'
                else:
                    end = '10年'
                index_end = i
                box_length_end = len(text)

    # found id
    id_result = id_reg.findall(infomation)
    if len(id_result):
        driver_id = id_result[0]
        index_id = infomation.index(driver_id)

        is_x = infomation[index_id+len(driver_id)]
        if  is_x == 'X' or is_x == 'x':
            driver_id = driver_id + 'X'

    # found gender
    if '性别' in infomation:
        index_ = infomation.index('性别')
        candidate_str = infomation[index_+2:index_+2+5]
    elif driver_id:
        # index_ = infomation.index(driver_id)
        candidate_str = infomation[index_id+len(driver_id) : index_id+len(driver_id)+15]
    
    if '男' in candidate_str:
        gender = '男'
    elif '女' in candidate_str:
        gender = '女'
    
    if not gender and driver_id:
        if len(driver_id) == 18:
            gender = '男' if int(driver_id[-2]) % 2 else '女'
        else:
            gender = '男'

    # found name
    if driver_id:
        candidate_str = infomation[index_id+len(driver_id)+1 : index_id+len(driver_id)+10]
        candidate_split = candidate_str.split(',')
        if '姓名' in candidate_str:
            index_temp = candidate_str.index('姓名')
            name = candidate_str[index_temp : ].split(',')[1]
        elif len(candidate_split[0]) < 3:
            if len(candidate_split[1]) < 2:
                name = candidate_str.split(',')[2]
            else:
                name = candidate_str.split(',')[1]
        else:
            name = candidate_str.split(',')[0]
            if len(name) > 3:
                name = name[-3:]
    else:
        if '姓名' in infomation:
            index_temp = infomation.index('姓名')
            name = infomation[index_temp : ].split(',')[1]

    # found address
    index_address = lengths.index(max(lengths[5:]))
    address = texts[index_address]
    address_y0 = boxes[index_address][1]
    for i in range(3):
        if index_address+i+1 < len(boxes):
            x0, y0, x1, y1 = boxes[index_address+i+1]
            text = texts[index_address+i+1]
            if y0 - address_y0 < 2*avg_h:
                if '男' not in text and '中国' not in text:
                    address += text

    # found class
    if '准驾车型' in infomation:
        idx = infomation.index('准驾车型')
        # class_ = infomation[idx:].split(',')[1]
        index_class = infomation[:idx].count(',') + 1
        class_ = texts[index_class]
        # check right one
        x0, y0, x1, y1 = boxes[index_class]
        xx0, yy0, xx1, yy1 = boxes[index_class+1]
        if xx0 > x0 and len(texts[index_class+1])<=2:
            class_ += texts[index_class+1]

    elif first and end:
        the_most_right_x = [x[2] for x in boxes[index_first+1 : index_end]]
        idx = the_most_right_x.index(max(the_most_right_x))
        index_class = index_first + 1 + idx
        class_ = texts[index_class]
        # check left one
        x0, y0, x1, y1 = boxes[index_class]
        xx0, yy0, xx1, yy1 = boxes[index_class-1]
        interval = abs(x0 - xx1)
        if x0 > xx0 and len(texts[index_class-1])<=2 and interval < (y1-y0):
            class_ = texts[index_class-1] + class_

    if class_:
        new_class = ''
        for char in class_:
            if ord(char) < 48 or ord(char) > 122:
                new_class += '1'
            else:
                new_class += char
        class_ = new_class

    # found logo
    # if brith:
    #     if box_length_brith > 10:
    #         index_logo_1 = index_brith - 1
    #     else:
    #         index_logo_1 = index_brith - 2
    #     text = texts[index_logo_1]
    #     logo += text

    #     index_logo_2 = index_brith + 1
    #     text = texts[index_logo_2]
    #     if '市公' in text or '公安' in text or '安局' in text or '局交' in text:
    #         text = '市公安局交'
    #     logo += text

    # if first and index_first+1<len(texts):
    #     index_logo_3 = index_first + 1
    #     text = texts[index_logo_3]
    #     if '支队' in text or '通' in text:
    #         text = '通警察支队'
    #     logo += text

    # validation dates
    if first and start:
        if not end or end in ['长期', '6年', '10年']:
            first, start = date_cross_validation_first_start(first, start)
        else:
            first, start, end = date_cross_validation(first, start, end)

    if not class_:
        class_ = 'C1'

    if class_ == 'C3' or class_ == '11':
        class_ = 'C1'

    if first and not start:
        start = first

    if start and not first:
        first = start
        
    if first and start and not end:
        if '长期' in infomation:
            end = '长期'
        elif '6年' in infomation:
            end = '6年'
        elif '10年' in infomation:
            end = '10年'
        else:
            end = start[:2] + str(6+int(start[2:4])) + start[4:]

    # print(driver_id)
    # print(name)
    # print(gender)
    # print(nation)
    # print(class_)
    # print(brith)
    # print(first)
    # print(start)
    # print(end)
    # print(address)
    # print(logo)

    return True, {
        'driver_id': driver_id,
        'name': name,
        'gender': gender,
        'nation': nation,
        'class': class_,
        'address': address,
        'brith': brith,
        'first': first,
        'start': start,
        'end': end
    }


def date_validate(date):
    year, month, day = date.split('-')

    if len(year) == 4:
        if int(year[0]) > 2:
            if year[0] == '7':
                year = '1' + year[1:]
            else:
                year = '2' + year[1:]
        if int(year[0]) == 2 and int(year[1]) > 0:
            year = '20' + year[2:]
        if int(year[0]) == 1:
            year = '19' + year[2:]
        if year[:2] == '20':
            if int(year[2]) > 2:
                if int(year[2]) == 7:
                    year = '201' + year[-1]
                else:
                    year = '200' + year[-1]

    if len(month) == 2:
        if int(month[0]) > 1:
            if int(month[0]) == 7:
                month = '1' + month[-1]
            else:
                month = '0' + month[-1]
        if month[0] == '1' and int(month[1]) > 2:
            if int(month[1]) == 7:
                month = '11'
            else:
                month = '10'

    if len(day) == 2:
        if day[0] == '3' and int(day[1]) > 1:
            if day[1] == '7':
                day = '31'
            else:
                day = '30'
        elif int(day[0]) > 3:
            if day[0] == '7':
                day = '1' + day[-1]
            else:
                day = '0' + day[-1]

    return year + '-' + month + '-' + day


def date_cross_validation(first, start, end):
    first = '%04d-%02d-%02d' % tuple(map(int, first.split('-')))
    start = '%04d-%02d-%02d' % tuple(map(int, start.split('-')))
    end = '%04d-%02d-%02d' % tuple(map(int, end.split('-')))

    f_year, f_month, f_day = first.split('-')
    s_year, s_month, s_day = start.split('-')
    e_year, e_month, e_day = end.split('-')
    
    monthes = [f_month, s_month, e_month]
    monthes_int = list(map(int, monthes))
    month_set = set(monthes)
    if len(month_set) == 3:
        return first, start, end
    elif len(month_set) == 2:
        counter = np.bincount(monthes_int)
        mode = np.argmax(counter)
        mode = '%02d' % mode
        f_month = mode
        s_month = mode
        e_month = mode
    
    if f_month == s_month == e_month:
        if not f_day==s_day==e_day:
            suffix = f_day[1] + s_day[1] + e_day[1]
            if '1' in suffix and '7' in suffix:
                f_day = f_day[0] + '1'
                s_day = s_day[0] + '1'
                e_day = e_day[0] + '1'

            dayes = [f_day, s_day, e_day]
            if len(set(dayes)) == 2:
                dayes_int = list(map(int, dayes))
                counter = np.bincount(dayes_int)
                mode = np.argmax(counter)
                mode = '%02d' % mode
                f_day = mode
                s_day = mode
                e_day = mode

    first = '-'.join([f_year, f_month, f_day])
    start = '-'.join([s_year, s_month, s_day])
    end = '-'.join([e_year, e_month, e_day])
    return first, start, end


def date_cross_validation_first_start(first, start):
    first = '%04d-%02d-%02d' % tuple(map(int, first.split('-')))
    start = '%04d-%02d-%02d' % tuple(map(int, start.split('-')))
    f_year, f_month, f_day = first.split('-')
    s_year, s_month, s_day = start.split('-')
    if f_month == s_month and f_day != s_day:
        if f_day[1] == '1' and s_day[1] == '7':
            s_day = f_day
        elif f_day[1] == '7' and s_day[1] == '1':
            f_day = s_day
    first = '-'.join([f_year, f_month, f_day])
    start = '-'.join([s_year, s_month, s_day])

    return first, start


if __name__ == '__main__':
    root = r'D:\xiashu\OCR\cl253_demo\regular'
    for f in os.listdir(root):
        fn = os.path.join(root, f)
        if f[-4:] == '.jpg':
            continue
        with open(fn, 'r', encoding='utf-8') as f:
            data = f.readlines()
            # data = [x.strip('\n').split(' ')[0] for x in data]
            # text = ','.join(data)
            # print(text)
            # print(id_reg.findall(text))
            texts = [x.strip('\n').split(' ')[0] for x in data]
            boxes_s = [x.strip('\n').split(' ')[1] for x in data]
            boxes = []
            for x in boxes_s:
                x0, y0, x1, y1 = x.split(',')
                boxes.append((int(x0), int(y0), int(x1), int(y1)))
            text_only_information_handle(texts, boxes)
            # print(boxes)
        print('')