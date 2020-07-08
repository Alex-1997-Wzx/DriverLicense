import os

PROVINCE = 2
CITY = 4
COUNTRY = 6
STREET = 8
VILLAGE = 10

PRE_CLASS = [PROVINCE, CITY, COUNTRY, STREET]
ALL_CLASS = [PROVINCE, CITY, COUNTRY, STREET, VILLAGE]

def build_addr_list(filename):
    fn = open(filename, encoding='utf-8')
    data = fn.read()
    fn.close()

    texts = data.split(';')
    print('found address: ', len(texts))

    c_4 = []
    c_5 = []

    length = len(texts)
    for i, text in enumerate(texts):
        if i < 3:
            continue
        if i > (length-2):
            break
        try:
            t = text.split('VALUES (')[-1][:-1]
            t = t.split(', ')
        except Exception as e:
            print(str(e))
            print(text)
            continue

        # print('-------------------------------------------------------')

        for k, addr in enumerate(t):
            if k in ALL_CLASS:
                addr = addr.strip('\'')
                addr = addr.strip(' ')
                # print(item)
                if k == VILLAGE:
                    c_5.append(addr)
                else:
                    if addr not in c_4:
                        c_4.append(addr)
    return c_4, c_5


def load_province_city_country_street():
    with open('./material/province_city_country_street.txt', encoding='utf-8') as f:
        data = f.read()
    return data.split('\n')

def load_village():
    with open('./material/village.txt', encoding='utf-8') as f:
        data = f.read()
    return data.split('\n')


if __name__ == '__main__':
    c4, c5 = build_addr_list('province_city_county_street_village.sql')
    print('got return')
    print(len(c4))
    # print('village')
    print(len(c5))
    with open('./material/province_city_country_street.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(c4))
    with open('./material/village.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(c5))

