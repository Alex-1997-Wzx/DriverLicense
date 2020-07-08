#encoding=utf-8
import os
import sys
import random
import json
import address_5_class
from datetime import datetime
from fake_id_generator.validator import validator


class FakeGanerator():
    # 驾驶类型, 15种
    CLASSES = tuple('A1 A2 A3 B1 B2 C1 C2 C3 C4 D E F M N P'.split())
    CHINESE = ''
    SURNAME_S = '赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程邢裴陆荣翁荀羊於惠甄曲家封芮羿储靳邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓屠蒙池乔阴能苍双闻莘党翟谭贡姬申扶堵冉雍卻桑桂牛寿通边扈燕冀浦尚农温别庄晏柴瞿阎慕连茹习宦艾鱼容向古易慎廖庾终暨居衡步都耿满弘匡国文寇禄阙东欧沃利蔚越夔隆师巩聂晁敖融訾辛那简饶空曾毋沙养鞠丰巢关相查游竺盖'
    SURNAME_D = '司马上官欧阳夏侯诸葛东方皇甫尉迟公冶淳于太叔申屠公孙轩辕令狐钟离宇文长孙慕容司徒端木拓跋百里东郭西门南宫'

    CURRENT_YEAR = int(datetime.today().year)

    def __init__(self, chinese_txt='./material/chinese_2000.txt', addr_txt_path='./material/address_json', top=None):
        self.chinese_txt = chinese_txt
        self.addr_txt_path = addr_txt_path

        self.class_count = 0

        if top is not None and int(top)>100:
            with open('./material/chinese_top_3000.txt', encoding='utf-8') as f:
                self.CHINESE = f.readline()[:top]

        self.firstname_top_200 = []
        with open('./material/names/firstname_top_200.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.strip('\n')
                self.firstname_top_200.append(line)

        self.sencondname_top_5000 = []
        with open('./material/names/secondname_top_5000.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()
            for line in data:
                line = line.strip('\n')
                self.sencondname_top_5000.append(line)


    def generate_class(self):
        ''' 随机生成驾驶证类别 '''
        if self.class_count % 2:
            return random.choice(self.CLASSES)
        else:
            return ''.join(random.sample(self.CLASSES, k=2))

    def generate_name(self):
        ''' 随机生成名字 '''
        # load chinese
        if not self.CHINESE:
            with open(self.chinese_txt, encoding='utf-8') as f:
                self.CHINESE = f.readline()
        # random choice a single or double chinese surname
        select = random.choices(('single', 'double'), weights=(0.99, 0.01))[0]
        if select == 'single':
            surname = random.choice(self.SURNAME_S)
        else:
            num = random.randint(0, len(self.SURNAME_D)-1)
            surname = self.SURNAME_D[2*num: 2*num+2]
        # random choice name, length 1 or 2
        name_len = random.choices((2, 1), (0.8, 0.2))[0]
        name = ''.join(random.sample(self.CHINESE, name_len))

        return surname + name

    def generate_name_popular(self):
        fn = random.choice(self.firstname_top_200)
        sn = random.choice(self.sencondname_top_5000)
        return fn + sn

    def generate_date(self):
        ''' 随机生成日期, 1960年之后的 '''
        year = random.randint(1960, self.CURRENT_YEAR)
        mouth = random.randint(1, 12)
        if mouth in [1, 3, 5, 7, 8, 10, 12]:
            day = random.randint(1, 31)
        elif mouth in [4, 6, 9, 11]:
            day = random.randint(1, 30)
        else:
            day = random.randint(1, 28)
        date = '%04d-%02d-%02d' % (year, mouth, day)
        return date

    def add_date_year(self, old_date, year):
        ''' 在原有日期基础上加上n年, 输入old_date格式1990-10-10 '''
        y, m, d = old_date.split('-')
        y = int(y) + int(year)
        return str(y) + '-' + m + '-' + d

    def generate_id(self):
        ''' 随机生成18位的驾驶证编号 '''
        # return str(random.randint(100000000000000000, 999999999999999999))
        return validator.fake_id()

    def generate_address(self):
        ''' 随机生成省+市+县+随机中文的地址 '''
        if not hasattr(self, 'province_list'):
            # load address data
            with open(os.path.join(self.addr_txt_path, 'province.json'), 'rb') as f:
                self.province_list = json.load(f)
            with open(os.path.join(self.addr_txt_path,'city.json'), 'rb') as f:
                self.city_dict = json.load(f)
            with open(os.path.join(self.addr_txt_path, 'country.json'), 'rb') as f:
                self.country_dict = json.load(f)
            with open(os.path.join(self.addr_txt_path, 'town.json'), 'rb') as f:
                self.town_dict = json.load(f)
            # load chinese
            if not self.CHINESE:
                with open(self.chinese_txt, encoding='utf-8') as f:
                    self.CHINESE = f.readline()

        province = ''
        province_id = ''
        city = ''
        city_id = ''
        country = ''
        country_id = ''
        town = ''
        town_id = ''

        # random choice province
        p = random.choice(self.province_list)
        province = p.get('name')
        province_id = p.get('id')

        # random choice city
        if self.city_dict.get(province_id):
            c = random.choice(self.city_dict[province_id])
            city = c.get('name')
            city_id = c.get('id')

        # random choice country
        if city_id and self.country_dict.get(city_id):
            ct = random.choice(self.country_dict[city_id])
            country = ct.get('name')
            country_id = ct.get('id')
        
        # random choice town
        if country_id and self.town_dict.get(country_id):
            t = random.choice(self.town_dict[country_id])
            town = t.get('name')
            town_id = t.get('id')
        
        if city == '市辖区':
            city = ''
        if '特殊镇' in town:
            town = ''
        # random generate road or village
        suffix = random.choices(population=['路', '村', '组', '街道', '弄'],
                                weights=[0.6, 0.2, 0.05, 0.1, 0.05])
        suffix = suffix[0]

        village_len = random.randint(1, 4)
        village = random.sample(self.CHINESE, village_len)
        village = ''.join(village)

        number = str(random.randint(1, 9999))
        return province + city + country + town + village + suffix + number + '号'

    
    def generate_address_c5(self, length=19):
        ''' 根据省、市、县、街道、乡村5个等级生成随机地址，格式为：
            随机省（市/县/街道）+ 乡村 '''
        if not hasattr(self, 'c4'):
            self.index = 0
            print('loading province_city_country_street...')
            self.c4 = address_5_class.load_province_city_country_street()
            print('shuffle province_city_country_street...')
            random.shuffle(self.c4)
            print('loading village...')
            self.c5 = address_5_class.load_village()
            print('shuffle village...')
            random.shuffle(self.c5)

        len_c5 = len(self.c5)
        # 随机选取一个省或市或区或县或街道
        address = random.choice(self.c4)
        while self.index < len_c5 and len(address) + len(self.c5[self.index]) <= length:
            address += self.c5[self.index]
            self.index += 1

        if (length-len(address)) >= 6:
            address += str(random.randint(1, 999))
            suffix = random.choices(population=['号', '路', '组',  '弄'],
                                weights=[0.5, 0.3, 0.1, 0.1])
            address += suffix[0]

        self.index += 1
        if self.index >= len_c5:  # 一个循环结束，重头开始
            self.index = 0
        return address


if __name__ == '__main__':
    faker = FakeGanerator()
    # addr = faker.generate_address()
    # for addr in addrs:
    addr = faker.generate_name_popular()
    print(addr)
    addr = faker.generate_class()
    print(addr)
    # date = faker.generate_date()
    # print(date)
    # for i in range(100):
    #     print(faker.generate_name())