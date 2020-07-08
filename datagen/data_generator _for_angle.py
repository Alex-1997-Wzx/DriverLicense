# encoding=utf-8
# data_generator.py
# 批量生成驾驶证图片
# 使用方式: python data_generator.py --help
import os
import sys
import json
import base64
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from faker import FakeGanerator
from test_template import randon_morphology

from chinese_name import makeName

BASELINE = random.randint(10000000000000, 100000000000001)

# 配置
parser = argparse.ArgumentParser()
# # 生成哪些图片？
# parser.add_argument('-g', '--gen',
#                     default='all',
#                     nargs='+',
#                     choices=['original', 'address', 'name', 'date', 'id', 'all'],
#                     help='选择要生成的图片：original-原图, address-地址栏，name-姓名栏，date-日期栏，id-驾驶证id栏，all-所有 (default: %(default)s)')
# 是否生成原始图片？
parser.add_argument('--image', dest='image', action='store_true', help ='是否生成原始图片？')
parser.add_argument('--no-image', dest='image', action='store_false', help ='是否生成原始图片？')
parser.set_defaults(image=True)
# 是否生成cut图片？
parser.add_argument('--cut', dest='cut', action='store_true', help ='是否生成cut图片')
parser.add_argument('--no-cut', dest='cut', action='store_false', help ='是否生成cut图片')
parser.set_defaults(cut=False)
# 是否生成英文标准信息？
parser.add_argument('--english', dest='english', action='store_true', help ='是否生成英文标准信息')
parser.add_argument('--no-english', dest='english', action='store_false', help ='是否生成英文标准信息')
parser.set_defaults(english=True)
# 生成多少张图片？
parser.add_argument('-a', '--amount', type=int, default=5000, help='要生成的图片数量')
# 生成多少张保存一次？
parser.add_argument('--saveInterval', type=int, default=100, help='每生成多少张图片存一次硬盘，提高图片保存效率')
# 输出路径？
parser.add_argument('--output', type=str, default='./gen', help="path to save generated data")
# 背景图片路径
parser.add_argument('--background_dir', type=str, default='./material/background', help="path of background image")
# 是否需要加入背景？
parser.add_argument('--background', dest='background', action='store_true', help ='是否需要加入背景？')
parser.add_argument('--no-background', dest='background', action='store_false', help ='是否需要加入背景？')
parser.set_defaults(background=True)
# 是否需要标注信息？
parser.add_argument('--label', dest='label', action='store_true', help ='是否输出位置标注信息')
parser.add_argument('--no-label', dest='label', action='store_false', help ='是否输出位置标注信息')
parser.set_defaults(label=True)
# 是否加入随机位置波动？
parser.add_argument('--position-bias', dest='pbias', action='store_true', help ='是否加入位置的随机扰动')
parser.add_argument('--no-position-bias', dest='pbias', action='store_false', help ='是否加入位置的随机扰动')
parser.set_defaults(pbias=True)
# 是否加入随机模糊处理？
parser.add_argument('--blur', dest='blur', action='store_true', help ='是否需要模糊化处理')
parser.add_argument('--no-blur', dest='blur', action='store_false', help ='是否需要模糊化处理')
parser.set_defaults(blur=True)
# 是否加入椒盐噪声？
parser.add_argument('--salt', dest='salt', action='store_true', help ='是否加入椒盐噪声')
parser.add_argument('--no-salt', dest='salt', action='store_false', help ='是否加入椒盐噪声')
parser.set_defaults(salt=True)
# 是否加入透视变化？
parser.add_argument('--perspective', dest='perspective', action='store_true', help ='是否加入透视变化')
parser.add_argument('--no-perspective', dest='perspective', action='store_false', help ='是否加入透视变化')
parser.set_defaults(perspective=True)
# 是否输出标注红框？
parser.add_argument('--redbox', dest='redbox', action='store_true', help ='是否绘制红色标注框，用于验证标注信息是否准确')
parser.add_argument('--no-redbox', dest='redbox', action='store_false', help ='是否绘制红色标注框，用于验证标注信息是否准确')
parser.set_defaults(redbox=False)
# 是否输出所有出现的中文字符集，输出为txt
parser.add_argument('--build-chinese', dest='chinese', action='store_true', help ='是否输出所用到的中文集（文本）')
parser.add_argument('--no-build-chinese', dest='chinese', action='store_false', help ='是否输出所用到的中文集（文本）')
parser.set_defaults(chinese=True)
opt = parser.parse_args()
print(opt)


# build output path
if not os.path.exists(opt.output):
    os.mkdir(opt.output)
txtDir = os.path.join(opt.output, 'labels')
if not os.path.exists(txtDir):
    os.mkdir(txtDir)
# gen_items = opt.gen if opt.gen!='all' or 'all' not in opt.gen else ['original', 'address', 'name', 'date', 'id']
if opt.image:
    output_dir = os.path.join(opt.output, 'original')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
if opt.cut:
    cut_dir = os.path.join(opt.output, 'cut')
    if not os.path.exists(cut_dir):
        os.mkdir(cut_dir)

# 对比度及亮度调节
def adjust_contast_brightness(img):
    alpha = random.randint(90,110) * 0.01
    beta = random.randint(-10,10)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

# 生成椒盐噪声
def img_salt_pepper_noise(src, percetage=0.01):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(2, src.shape[0] - 3)
        randY = random.randint(2, src.shape[1] - 3)
        if random.randint(0, 1) == 0:
            color = (0, 0, 0)
        else:
            color = (255, 255, 255)
        NoiseImg[randX, randY] = color
        choice = random.choice([0, 1, 2])
        if choice == 0:
            NoiseImg[randX + 1, randY] = color
        elif choice == 1:
            NoiseImg[randX, randY+1] = color
    return NoiseImg

# 获取随机透视变换矩阵
def get_perspective_matrix(img, points):
    h, w = img.shape[:2]
    h_bias_left = random.randint(2, 12) * h // 100
    h_bias_right = random.randint(2, 12) * h // 100
    w_bias_left = random.randint(2, 12) * w // 100
    w_bias_right = random.randint(2, 12) * w // 100

    # 左图中画面中的点的坐标 四个
    pts1 = np.float32(points)
    # 变换到新图片中，四个点对应的新的坐标 一一对应
    # 随机选择两个点进行透视变换
    seed = random.randint(0, 3)
    new_ps = []
    # for i, xy in enumerate(points):
        # if i == seed or (seed+1)%4 == i:
    if seed == 0:
        new_ps.append((points[0][0]+random.randint(10, 30), points[0][1] + h_bias_left))
        new_ps.append((points[1][0]-random.randint(10, 30), points[1][1] + h_bias_right))
        new_ps.append(points[2])
        new_ps.append(points[3])
    elif seed == 1:
        new_ps.append(points[0])
        new_ps.append((points[1][0] - w_bias_left, points[1][1] + random.randint(10, 40)))
        new_ps.append((points[2][0] - w_bias_right, points[2][1] - random.randint(10, 40)))
        new_ps.append(points[3])
    elif seed == 2:
        new_ps.append(points[0])
        new_ps.append(points[1])
        new_ps.append((points[2][0]-random.randint(10, 30), points[2][1] - h_bias_left))
        new_ps.append((points[3][0]+random.randint(10, 30), points[3][1] - h_bias_right))
    else:
        new_ps.append((points[0][0]+w_bias_left, points[0][1] + random.randint(10, 40)))
        new_ps.append(points[1])
        new_ps.append(points[2])
        new_ps.append((points[3][0]+w_bias_right, points[3][1] - random.randint(10, 40)))

    pts2 = np.float32(new_ps)

    # 生成变换矩阵
    M = cv2.getPerspectiveTransform(pts1,pts2)
    # 进行透视变换
    # dst = cv2.warpPerspective(img,M,(300,300))
    return M


# data generate
def generate_driver_card():
    ''' 批量生产虚拟驾驶证 '''
    # baseline = 2000000
    # ratio_width = 755 / 1133
    # ratio_height = 529 / 794

    # create font object with the font file and specify desired size
    font_song1 = ImageFont.truetype('./material/font/song1.ttf', size=48)
    font_song2 = ImageFont.truetype('./material/font/song2.ttf', size=45)
    font_song3 = ImageFont.truetype('./material/font/song3.ttf', size=44)
    font_id = ImageFont.truetype('./material/font/jishi4.ttf', size=36)
    font_birth = ImageFont.truetype('./material/font/jishi4.ttf', size=42)
    font_msyh = ImageFont.truetype('./material/font/msyh.ttf', size=45)
    font_xinshi = ImageFont.truetype('./material/font/xinshi.ttf', size=38)

    # font list
    print('load fonts...')
    font_song1_ls = [ImageFont.truetype('./material/font/song1.ttf', size=x) for x in range(46, 65)]
    font_song3_ls = [ImageFont.truetype('./material/font/song3.ttf', size=x) for x in range(40, 49)]
    font_id_ls = [ImageFont.truetype('./material/font/jishi4.ttf', size=x) for x in range(32, 41)]
    font_birth_ls = [ImageFont.truetype('./material/font/jishi4.ttf', size=x) for x in range(34, 43)]

    # other fonts
    # other_fonts = []
    for f_ in os.listdir('./material/font/random_font'):
        fn_ = os.path.join('./material/font/random_font', f_)
        for x_ in range(32, 65):
            imfont = ImageFont.truetype(fn_, size=x_)
            if imfont is None:
                print('error font: ' , 'fn')
                continue
            if 46 <= x_ <= 65:
                font_song1_ls.append(imfont)
            if 40 <= x_ <= 49:
                font_song3_ls.append(imfont)
            if 32 <= x_ <= 41:
                font_id_ls.append(imfont)
            if 34 <= x_ <= 43:
                font_birth_ls.append(imfont)

    X, Y = 0, 1
    BASE_POS, POS, FONT, FILL, TEXT, LABEL = 0, 1, 2, 3, 4, 5

    # 驾驶本基本数据，分别为：基础坐标、实际坐标、字体、颜色、文本、标注信息
    items = {
        'name': [(240, 240), [280, 240], font_song1_ls, (75, 75, 75), "王宏", None],
        'id': [(477, 172), [477, 172], font_id_ls, (80, 80, 80), "374562876538274561", None],
        # 'class': [(608, 608), [608, 608], font_song3, (80, 80, 80), "C1", None],
        'birth': [(497, 452), [497, 452], font_birth_ls, (80, 80, 80), "1990-08-08", None],
        'first': [(544, 524), [544, 524], font_birth_ls, (80, 80, 80), "2016-10-23", None],
        'start': [(291, 680), [291, 680], font_birth_ls, (80, 80, 80), "2016-10-23", None],
        'end': [(598, 680), [598, 680], font_birth_ls, (80, 80, 80), "2022-10-23", None],
        # address2 增加一列，用于处理过长地址
        'address2': [(220, 374), [220, 374], font_song3_ls, (80, 80, 80), "", None],
        'address': [(220, 304), [220, 304], font_song3_ls, (80, 80, 80), "重庆市渝北区龙景路158号", None],
    }

    # load template
    print('tempaltes load...')
    templatesName = []
    templateStrings = {}
    # template_width = 1133
    # template_height = 794
    for tem in os.listdir('./material/templates'):
        temFn = os.path.join('./material/templates', tem)
        if not os.path.isfile(temFn):
            continue
        templatesName.append(temFn)
        # srcimage = Image.open(temFn)
        # srcimage = srcimage.convert('RGB')
        # srcimage = srcimage.resize((1133, 794), Image.BOX)

        # buffered = BytesIO()
        # srcimage.save(buffered, format="JPEG")
        # img_encode = base64.b64encode(buffered.getvalue())
        # image_string = BytesIO(base64.b64decode(img_encode))
        # templateStrings.append(image_string)

    # load background
    bg_images = []
    if opt.background:
        print('load background images...')
        for f in os.listdir(opt.background_dir):
            fn = os.path.join(opt.background_dir, f)
            im_ = cv2.imread(fn)
            if im_ is not None:
                bg_images.append(im_)

    # load faker generator
    print('load fake generator...')
    fakeGen = FakeGanerator()

    imagesBatch = {'original':[], 'cut':[]}
    labels = []  # label = [filename, [()]]

    from add_face import AddFace
    addFaceObj = AddFace()

    for ep in range(opt.amount):
        if ((ep+1) % 100 == 0):
            print("images {}/{}".format(ep+1, opt.amount))

        # 其他固定文本的信息
        others = [
            ["中华人民共和国机动车驾驶证", [(235,78), (900,78), (900,132), (235,132)]],
            ["证号", [(345,170), (436,170), (436,210), (345,210)]],
            ["姓名", [(87,241), (154,241), (154,270), (87,270)]],
            ["性别", [(536,241), (602,241), (602,270), (536,270)]],
            ["国籍", [(715,241), (784,241), (784,270), (715,270)]],
            ["中国", [(849,227), (940,227), (940,272), (849,272)]],
            ["住址", [(88,310), (152,310), (152,340), (88,340)]],
            ["出生日期", [(324,451), (452,451), (452,482), (324,482)]],
            ["初次领证日期", [(322,527), (516,527), (516,557), (322,557)]],
            ["准驾车型", [(321,605), (455,605), (455,635), (321,635)]],
            ["有效期限", [(97,687), (225,687), (225,716), (97,716)]],
            ["至", [(531,687), (570,687), (570,719), (531,719)]],
            ["DrivingLicenseofthePeople'sRepublicofChina", [(241,136), (891,136), (891,164), (241,164)]],
            ["Name", [(85,273), (160,273), (160,295), (85,295)]],
            ["Sex", [(544,272), (590,272), (590,296), (544,296)]],
            ["Nationality", [(717,273), (847,273), (847,298), (717,298)]],
            ["Address", [(85,345), (186,345), (186,368), (85,368)]],
            ["DateofBirth", [(323, 485), (474,485), (474,508), (323,508)]],
            ["DateofFirstIssue", [(322,560), (531,560), (531,584), (322,584)]],
            ["Class", [(329,639), (395,639), (395,658), (329,658)]],
            ["ValidPeriod", [(95,714), (239,714), (239,739), (95,739)]],
            # 下面的文本会变化，根据模板不同作修改
            ["男", [(640,228), (686, 228), (686,273), (640,273)]],
            ["red_0", [(88,447), (307,447), (307,500), (88,500)]],
            ["red_1", [(88,532), (307,532), (307,582), (88,582)]],
            ["red_2", [(88,610), (307,610), (307,660), (88,660)]],
            ["C1", [(592,602), (652,602), (652,643), (592,643)]]  # 出现A1B1这种4个字符的，y轴向两边分别扩充30个像素点
        ]

        # 是否加入随机扰动
        if opt.pbias:
            # v_bias 控制上下偏移
            h_bias = random.randint(-10, 10)
            v_bias = random.randint(-12, 12)
            for key, val in items.items():
                val[POS][X] = val[BASE_POS][X] + h_bias
                val[POS][Y] = val[BASE_POS][Y] + v_bias

        # 随机生成一个地址
        address = fakeGen.generate_address_c5(length=random.randint(15, 25))
        address = address.replace('?', '')
        # 若地址长度大于18，则换行
        if len(address) > 18:
            items['address'][TEXT] = address[:18]
            items['address2'][TEXT] = address[18:]
        else:
            items['address'][TEXT] = address
            items['address2'][TEXT]=''
        items['name'][TEXT] = fakeGen.generate_name() if ep % 2 else makeName()
        items['id'][TEXT] = fakeGen.generate_id()  # 生成有效身份证
        # items['class'][TEXT] = fakeGen.generate_class()
        # items['birth'][TEXT] = fakeGen.generate_date()
        birth = items['id'][TEXT][6:14]
        items['birth'][TEXT] = birth[:4] + '-' + birth[4:6] + '-' + birth[6:]   # 出生日期根据身份证id生成
        items['first'][TEXT] = fakeGen.generate_date()  # 第一次领证日期
        items['start'][TEXT] = items['first'][TEXT]  # 有效起始日期与第一次领证日期一致
        seed_end = random.randint(0, 9)
        if 0 <= seed_end < 7:
            items['end'][TEXT] = fakeGen.add_date_year(items['start'][TEXT], random.choice([6, 10]))
        elif seed_end == 7:
            items['end'][TEXT] = '6年'
        elif seed_end == 8:
            items['end'][TEXT] = '10年'
        else:
            items['end'][TEXT] = '长期'

        # initialise the drawing context with the image object as background
        rd_filename = random.choice(templatesName)
        # 选择文件名称后，解析红章文本/性别/驾驶证类别
        tmp_fn = os.path.split(rd_filename)[-1]
        tmp_fn = os.path.splitext(tmp_fn)[0]
        org, cl, sex = tmp_fn.split('-')
        len_org = len(org)
        # 据统计，红章分行规则如下
        if len_org == 13:
            org_0, org_1, org_2 = org[:4], org[4:8], org[8:]  # 4,4,5
        elif len_org == 14:
            org_0, org_1, org_2 = org[:5], org[5:9], org[9:]  # 5,4,5
        elif len_org == 15:
            org_0, org_1, org_2 = org[:5], org[5:10], org[10:]  # 5,5,5
        elif len_org == 16:
            org_0, org_1, org_2 = org[:6], org[6:11], org[11:]  # 6,5,5
        elif len_org == 17:
            org_0, org_1, org_2 = org[:5], org[5:11], org[11:]  # 5,6,6
        elif len_org == 18:
            org_0, org_1, org_2 = org[:6], org[6:12], org[12:]  # 6,6,6
        elif len_org == 19:
            org_0, org_1, org_2 = org[:7], org[7:13], org[13:]  # 7,6,6
        elif len_org == 20:
            org_0, org_1, org_2 = org[:6], org[6:13], org[13:]  # 6,7,7
        elif len_org == 21:
            org_0, org_1, org_2 = org[:8], org[8:15], org[15:]  # 8,7,6
        else:
            print('红章换行规则解析错误: ', rd_filename)
            continue

        others[-1][0] = cl
        others[-2][0] = org_2
        others[-3][0] = org_1
        others[-4][0] = org_0
        others[-5][0] = sex

        if rd_filename not in templateStrings:
            srcimage = Image.open(rd_filename)
            srcimage = srcimage.convert('RGB')
            srcimage = srcimage.resize((1133, 794), Image.BOX)

            buffered = BytesIO()
            srcimage.save(buffered, format="JPEG")
            img_encode = base64.b64encode(buffered.getvalue())
            image_string = BytesIO(base64.b64decode(img_encode))
            templateStrings[rd_filename] = image_string

        image = Image.open(templateStrings[rd_filename])
        draw = ImageDraw.Draw(image)

        # get random fill
        rd_fill = random.randint(-75, 20)
        new_fill = (80 + rd_fill, ) * 3
        # draw the text on the background
        error = False
        for key, val in items.items():
            if key == 'end' and val[TEXT] in ['6年', '10年', '长期']:
                new_font = random.choice(font_song3_ls)
            else:
                new_font = random.choice(val[FONT])  # random select a font
            # new_font = random.choice(val[FONT])  # random select a font
            if key == 'name':
                val[TEXT] = ' '.join(list(val[TEXT]))
            try:
                xy, mask = draw.text(val[POS], val[TEXT], fill=new_fill, font=new_font)
            except OSError as e:
                print(str(e))
                error = True
                break
            x0, y0 = xy
            x1 = x0 + mask.size[0]
            y1 = y0 + mask.size[1]
            # x0, y0, x1, y1 = x0 - 5, y0 - 5, x1 + 5, y1 + 5
            val[LABEL] = (x0, y0, x1, y0, x1, y1, x0, y1)

        if error:
            continue

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        # add face
        image = addFaceObj.add_face_to_img(image)

        if opt.background:
            bgImg = random.choice(bg_images)
            angle = random.randint(-7, 7)
            height, width = bgImg.shape[:2]
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            bgRotate = cv2.warpAffine(bgImg, M, (width, height))
            
            src_h, src_w = image.shape[:2]

            # 设计随机比例，将驾驶证图片嵌入背景图中
            ratio = random.randint(65, 90) / 100
            if width/height < src_w/src_h:  # 
                target_w = int(width * ratio)
                target_h = int(src_h * target_w / src_w)
            else:
                target_h = int(height * ratio)
                target_w = int(src_w * target_h / src_h)
            x0 = (width - target_w) // 2
            y0 = (height - target_h) // 2
            x1 = x0 + target_w
            y1 = y0 + target_h

            if y0 <= 0 or y1 >= height:
                print('height out of range..')
                continue
            if x0 <= 0 or x1 >= width:
                print('width out of range..')
                continue

            inner = cv2.resize(image, (target_w, target_h))

            # resize后记录的标注信息也要跟着变
            for key, val in items.items():
                val[LABEL] = tuple([int(x * target_w / src_w) for x in val[LABEL]])
            for other in others:
                new_xy = []
                for x,y in other[1]:
                    new_xy.append((int(x * target_w / src_w), int(y * target_w / src_w)))
                other[1].clear()
                other[1].extend(new_xy)

            alpha_bg = 0.01 * random.randint(5, 35)
            alpha_dv = 1 - alpha_bg
            bgRotate[y0:y1, x0:x1,:] = (alpha_bg * bgRotate[y0:y1, x0:x1,:] + alpha_dv * inner[:,:,:])
            image = bgRotate
            # 计算原来标注位置在新图片中的绝对位置
            for key, val in items.items():
                new_pos = []
                for i, v in enumerate(val[LABEL]):
                    if i % 2:
                        new_pos.append(y0 + v)
                    else:
                        new_pos.append(x0 + v)
                val[LABEL] = tuple(new_pos)

            for other in others:
                new_xy = []
                for x, y in other[1]:
                    new_xy.append((x+x0, y+y0))
                other[1].clear()
                other[1].extend(new_xy)


            if opt.salt:  # 椒盐噪声
                if 1 == random.randint(0,3):
                    image = img_salt_pepper_noise(image, 0.005)

            if opt.blur and image.shape[0]>500 and image.shape[1]>500:
                seed = random.choice((0, 3, 5))
                if seed:
                    # 随机选择滤波方式
                    blur_method = random.choice([cv2.GaussianBlur, cv2.blur, cv2.medianBlur])
                    if blur_method is cv2.bilateralFilter:
                        image = blur_method(image, -1, random.randint(20, 200), random.randint(20, 200))
                    elif blur_method is cv2.GaussianBlur:
                        image = blur_method(image, (seed, seed), 0)
                    elif blur_method is cv2.blur:
                        image = blur_method(image, (seed, seed))
                    else:
                        image = blur_method(image, seed)

            # 截图每个词条并保存
            if opt.cut:
                rd_bias = random.randint(0, 3)
                for key, val in items.items():
                    # 'name': [(300, 240), [300, 240], font_song1, (75, 75, 75), "王宏", None],
                    if key == 'address2':
                        if not val[TEXT]:  # 地址栏第二栏如果没有数据，则跳过
                            continue
                    # 以下词条样本量重复太多，在此作特殊处理，每50张图片生成一次
                    if key in ['birth', 'first', 'start', 'end', 'id']:
                        if ep % 50 != 0:
                            continue
                    x0_, y0_, x1_ = val[LABEL][:3]
                    y1_ = val[LABEL][-1]
                    x0_, y0_, x1_, y1_ = x0_ - rd_bias, y0_ - rd_bias, x1_ + rd_bias, y1_ + rd_bias
                    img_ = image[y0_:y1_, x0_:x1_, :]
                    filename = val[TEXT] + '_%s.jpg' %  (BASELINE+ep)  # 注意：此处可能要改,地址换行后名字各不同
                    fn_ = os.path.join(opt.output, 'cut', filename)
                    imagesBatch['cut'].append((img_, fn_))

                if ep % 50 == 0:
                    for other in others:
                        # ["中华人民共和国机动车驾驶证", [(234,75), (900,75), (900,134), (234,134)]],
                        if other[0] == "DrivingLicenseofthePeople'sRepublicofChina":  # 太长，舍去
                            continue
                        x0_, y0_ = other[1][0]
                        x1_, y1_ = other[1][2]
                        x0_, y0_, x1_, y1_ = x0_ - rd_bias, y0_ - rd_bias, x1_ + rd_bias, y1_ + rd_bias
                        img_ = bgRotate[y0_:y1_, x0_:x1_, :]
                        filename = other[0] + '_%s.jpg' %  (BASELINE+ep)  # 注意：此处可能要改,地址换行后名字各不同
                        fn_ = os.path.join(opt.output, 'cut', filename)
                        imagesBatch['cut'].append((img_, fn_))

            if opt.perspective:
                ps = [(x0,y0), (x1,y0), (x1,y1), (x0,y1)]
                pM = get_perspective_matrix(image, ps)
                image = cv2.warpPerspective(image, pM, (width, height))
                
                # 透视变换后，标注坐标也要变化
                for key, val in items.items():
                    ps = val[LABEL]
                    old_points = [(ps[i], ps[i+1]) for i in range(0, 8, 2)]
                    old_points = np.array(old_points, dtype='float32')
                    old_points = np.array([old_points])
                    new_points = cv2.perspectiveTransform(old_points, pM)

                    points_by_warp = []
                    for i in range(4):
                        points_by_warp.append(int(new_points[0][i][0]))
                        points_by_warp.append(int(new_points[0][i][1]))
                    val[LABEL] = tuple(points_by_warp)

                for other in others:
                    old_points = np.array(other[1], dtype='float32')
                    old_points = np.array([old_points])
                    new_points = cv2.perspectiveTransform(old_points, pM)

                    new_xy = []
                    for i in range(4):
                        new_xy.append((int(new_points[0][i][0]), int(new_points[0][i][1])))
                    other[1].clear()
                    other[1].extend(new_xy)


            invM = cv2.invertAffineTransform(M)
            image = cv2.warpAffine(image, invM, (width, height))

            # 仿射变换后标注坐标点也要跟着变换
            for key, val in items.items():
                ps = val[LABEL]
                old_points = [(ps[i], ps[i+1]) for i in range(0, 8, 2)]
                old_points = np.array(old_points, dtype=np.int32)
                old_points = np.reshape(old_points, (4,1,2))
                new_points = cv2.transform(old_points, invM)

                points_by_warp = []
                for i in range(4):
                    points_by_warp.append(new_points[i][0][0])
                    points_by_warp.append(new_points[i][0][1])
                val[LABEL] = tuple(points_by_warp)

            for other in others:
                # old_points = list(other[1])
                old_points = np.array(other[1], dtype=np.int32)
                old_points = np.reshape(old_points, (4,1,2))
                new_points = cv2.transform(old_points, invM)

                new_xy = []
                for i in range(4):
                    new_xy.append((new_points[i][0][0], new_points[i][0][1]))
                other[1].clear()
                other[1].extend(new_xy)

        # 随机加入形态学变换
        image = randon_morphology(image)

        # 所有图片保存成固定尺寸1280 * 897
        old_h, old_w = image.shape[:2]
        scale_w = 1280 / old_w
        scale_h = 897 / old_h
        image = cv2.resize(image, (1280, 897))
        # resize后记录的标注信息也要跟着变
        for key, val in items.items():
            # val[LABEL] = tuple([int(x * scale_w) for x in val[LABEL]])
            new_label = []
            for idx in range(4):
                new_label.append(int(val[LABEL][2*idx] * scale_w))
                new_label.append(int(val[LABEL][2*idx+1] * scale_h))
            # val[LABEL].clear()
            val[LABEL] = tuple(new_label)
        for other in others:
            new_xy = []
            for x,y in other[1]:
                new_xy.append((int(x * scale_w), int(y * scale_h)))
            other[1].clear()
            other[1].extend(new_xy)


        if opt.redbox:
            for key, val in items.items():
                points = val[LABEL]
                x0,y0,x1,y1,x2,y2,x3,y3 = points
                cv2.line(image, (x0, y0), (x1, y1), (0, 0, 255))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.line(image, (x2, y2), (x3, y3), (0, 0, 255))
                cv2.line(image, (x3, y3), (x0, y0), (0, 0, 255))
            for i, other in enumerate(others):
                # ["中华人民共和国机动车驾驶证", [(234,75), (900,75), (900,134), (234,134)]],
                (x0,y0), (x1,y1), (x2,y2), (x3,y3) = other[1]
                if len(others)-1 == i:
                    temp_fn = os.path.split(rd_filename)[-1]
                    temp_fn = os.path.splitext(temp_fn)[0]
                    try:
                        origanization, cl, sex = temp_fn.split('-')
                    except Exception as e:
                        print(str(e))
                        print(temp_fn)
                    if len(cl) > 2:
                        x0 = x3 = x0 - 30
                        x1 = x2 = x1 + 30
                cv2.line(image, (x0, y0), (x1, y1), (0, 0, 255))
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255))
                cv2.line(image, (x2, y2), (x3, y3), (0, 0, 255))
                cv2.line(image, (x3, y3), (x0, y0), (0, 0, 255))

        if opt.image:
            filename = address + '_%s.jpg' %  (BASELINE+ep)
            originalFilename = os.path.join(opt.output, 'original', filename)
            imagesBatch['original'].append((image, originalFilename))

            # 在保存原图的情况下，如果选择了保存标注信息，则在此处控制保存
            if opt.label:
                text_label = []
                for key, val in items.items():
                    if key == 'address2':
                        if not val[TEXT]:  # 地址栏第二栏如果没有数据，则跳过
                            continue
                    # text_label.append(','.join(list(map(str,val[LABEL]))) + ',' + key + ',' + val[TEXT])
                    text_label.append(','.join(list(map(str,val[LABEL]))) + ',' + val[TEXT])
                for idx, other in enumerate(others):
                    if not opt.english:
                        if idx != (len(others)-1) and 'A'<=other[0][0]<='z':
                            continue
                    # text_label.append(','.join([str(x_) for xy_ in other[1] for x_ in xy_]) + ',' + 'other' + ',' + other[0])
                    text_label.append(','.join([str(x_) for xy_ in other[1] for x_ in xy_]) + ',' + other[0])
                label = '\n'.join(text_label)
                fn = address + '_%s.txt' %  (BASELINE+ep)
                txtFilename = os.path.join(opt.output, 'labels', fn)
                labels.append((label, txtFilename))

        # if not opt.background and 'address' in gen_items:
        #     x0, y0, x1 = items['address'][LABEL][:3]
        #     y1 = items['address'][LABEL][-1]
        #     addressImage = image[y0:y1, x0:x1, :]
        #     filename = address + '_%s.jpg' %  ep  # 注意：此处可能要改,地址换行后名字各不同
        #     addressFilename = os.path.join(opt.output, 'address', filename)
        #     imagesBatch['address'].append((addressImage, addressFilename))


        # 图片量达到saveInterval设定值后，一次性保存
        for key in ['original', 'cut']:
            if len(imagesBatch[key]) >= opt.saveInterval:
                for img, fn in imagesBatch[key]:
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(50, 90)]
                    try:
                        # img = adjust_contast_brightness(img)
                        # img = img_salt_pepper_noise(img, 0.01)
                        cv2.imencode('.jpg', img, encode_param)[1].tofile(fn)
                        # cv2.imwrite(fn, img)
                    except Exception as e:
                        print('error: ', fn)
                imagesBatch[key].clear()
        # 标注图片数达到saveInterval设定值后，一次性保存txt标注文件
        if len(labels) >= opt.saveInterval:
            for label, fn in labels:
                try:
                    with open(fn, 'w', encoding='utf-8') as f:
                        f.write(label)
                except Exception as e:
                    print('write txt error : ', fn)
            labels.clear()

    # 保存剩余图片
    for key in ['original', 'cut']:
        if len(imagesBatch[key]):
            for img, fn in imagesBatch[key]:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(50, 90)]
                try:
                    # img = adjust_contast_brightness(img)
                    # img = img_salt_pepper_noise(img, 0.01)
                    cv2.imencode('.jpg', img, encode_param)[1].tofile(fn)
                    # cv2.imwrite(fn, img)
                except Exception as e:
                    print('error: ', fn)
                if key == 'original' and opt.label:
                    txt_name = os.path.splitext(fn)[0] + '.txt'
            imagesBatch[key].clear()

    if len(labels):
        for label, fn in labels:
            try:
                with open(fn, 'w', encoding='utf-8') as f:
                    f.write(label)
            except Exception as e:
                print('write txt error : ', fn)
        labels.clear()

    print('Finished!')


if __name__ == '__main__':
    generate_driver_card()
    pass

