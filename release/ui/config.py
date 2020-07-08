#encoding=utf-8
import sys
import os
from configobj import ConfigObj


# 读取配置文件
cfgParser = ConfigObj('./ui/config.conf', encoding='utf-8')


AngleModel = str(cfgParser['files']['AngleModel'])
DetectModel = str(cfgParser['files']['DetectModel'])
RecognizeModel = str(cfgParser['files']['RecognizeModel'])
LexiconFile = str(cfgParser['files']['LexiconFile'])

TestPath = str(cfgParser['Recent']['TestPath'])

ImageData = int(cfgParser['Other']['ImageData'])


def update_settings():
    cfgParser['files']['AngleModel'] = AngleModel
    cfgParser['files']['DetectModel'] = DetectModel
    cfgParser['files']['RecognizeModel'] = RecognizeModel
    cfgParser['files']['LexiconFile'] = LexiconFile

    cfgParser['Recent']['TestPath'] = TestPath

    cfgParser.write()
