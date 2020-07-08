import os
import json
import time


def save_to_json(test_or_val,
                 results,
                 angle_model='',
                 detect_model='',
                 recog_model='',
                 lexicon_file='',
                 lot_name=''):
    data = dict(
        test_or_val = test_or_val,
        angle_model = angle_model,
        detect_model = detect_model,
        recog_model = recog_model,
        results = results)
    if not lot_name:
        lot_name = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    filename = os.path.join('./history', lot_name + '_' + str(len(results)) + '.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)