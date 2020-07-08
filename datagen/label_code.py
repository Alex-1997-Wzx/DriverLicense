''' 人工标注的数据格式转化, json转txt '''
import json
import os

ouput_dir = './data/labeled/txt'
img_path = './data/labeled/json'

def list_dir(path):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if file_path[-4:] == 'json':
            with open('%s' % file_path, 'r') as load_f:
                load_dict = json.load(load_f)
                targetfile = ouput_dir + '/{}.txt'.format(file[:-5])
                with open(targetfile, 'a', encoding='utf-8')as f:
                    for i in range(len(load_dict['shapes'])):
                        if len(load_dict["shapes"][i]['points']) == 2:
                            f.write(str(int(load_dict["shapes"][i]['points'][0][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][0][1])) + ',')
                            f.write(str(int(load_dict["shapes"][i]['points'][1][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][0][1])) + ',')
                            f.write(str(int(load_dict["shapes"][i]['points'][1][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][1][1])) + ',')
                            f.write(str(int(load_dict["shapes"][i]['points'][0][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][1][1])) + ',')
                            f.write(load_dict["shapes"][i]['label'] + '\n')
                        elif len(load_dict["shapes"][i]['points']) == 4:
                            f.write(str(int(load_dict["shapes"][i]['points'][0][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][0][1])) + ',')
                            f.write(str(int(load_dict["shapes"][i]['points'][1][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][1][1])) + ',')
                            f.write(str(int(load_dict["shapes"][i]['points'][2][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][2][1])) + ',')
                            f.write(str(int(load_dict["shapes"][i]['points'][3][0])) + ',' + str(
                                int(load_dict["shapes"][i]['points'][3][1])) + ',')
                            f.write(load_dict["shapes"][i]['label'] + '\n')
                        else:

                            print("标注坐标点过多")


if __name__ == '__main__':

    if not os.path.exists(ouput_dir):
        os.mkdir(ouput_dir)
    list_dir(img_path)
