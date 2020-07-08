import os
import sys
import copy
import json
import functools

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import cv2
import numpy as np

from ui import utils
from ui import config_widget
import ui.config as cfg

from ui.working_thread import TestThread
from ui.about_us import AboutUs
from ui.utils import contrast_one
from util.result_to_json import save_to_json
from util.chinese_opencv import paint_chinese_opencv


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('XIASHU-OCR 试用版')
        self.setWindowIcon(QtGui.QIcon('./ui/icons/xiashu.png'))

        # dialogs
        self.init_dialogs()

        # dock
        self.label_dock = QtWidgets.QDockWidget('Label', self)
        self.label_dock.setObjectName('Label')
        self.label_widget = QtWidgets.QListWidget()
        self.label_dock.setWidget(self.label_widget)

        self.file_search = QtWidgets.QLineEdit()
        self.file_search.setPlaceholderText('Search Filename')
        self.file_search.textChanged.connect(self.file_search_changed)
        self.file_list_widget = QtWidgets.QListWidget()
        self.file_list_widget.itemSelectionChanged.connect(self.file_select_changed)

        file_list_layout = QtWidgets.QVBoxLayout()
        file_list_layout.setContentsMargins(0, 0, 0, 0)
        file_list_layout.setSpacing(0)
        file_list_layout.addWidget(self.file_search)
        file_list_layout.addWidget(self.file_list_widget)
        self.file_dock = QtWidgets.QDockWidget('File List', self)
        self.file_dock.setObjectName('Files')
        file_list_widget = QtWidgets.QWidget()
        file_list_widget.setLayout(file_list_layout)
        self.file_dock.setWidget(file_list_widget)

        self.addDockWidget(Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.file_dock)

        # widgets
        self.path_edit = QtWidgets.QLineEdit()
        self.path_edit.setPlaceholderText('image directory select')
        self.open_button = utils.newButton('Open', 'open', self.open_directory)
        # self.open_button.clicked.connect(self.open_directory)
        self.load_button = utils.newButton('Load Model', 'model', self.ConfigDialog.show)
        # self.load_button.clicked.connect(self.load_model_trigger)
        self.start_button = utils.newButton('Start', 'start', self.run)
        # self.start_button.clicked.connect(self.run)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximum(100)
        
        path_widget = QtWidgets.QWidget()
        path_horizen = QtWidgets.QHBoxLayout()
        path_horizen.addWidget(self.open_button)
        path_horizen.addWidget(self.load_button)
        path_horizen.addWidget(self.start_button)
        path_horizen.addWidget(self.progress_bar)
        # h_spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        # path_horizen.addSpacerItem(h_spacer)
        path_layout = QtWidgets.QVBoxLayout()
        path_layout.addWidget(self.path_edit)
        path_layout.addLayout(path_horizen)

        # image label
        self.image_label = QtWidgets.QLabel()
        self.image_label.setStyleSheet('background-color: rgb(0, 0, 0);')
        self.image_label.setAlignment(Qt.AlignCenter)

        vboxlayout = QtWidgets.QVBoxLayout()
        vboxlayout.addLayout(path_layout)
        vboxlayout.addWidget(self.image_label)

        mainwidget = QtWidgets.QWidget()
        mainwidget.setLayout(vboxlayout)
        self.setCentralWidget(mainwidget)

        # actions
        action = functools.partial(utils.newAction, self)
        open_action = action('&Open', self.open_single_file)
        open_dir_action = action('&Open Dir', self.open_directory)
        save_action = action('&Save', self.save_file)
        save_as_action = action('&Save As', self.save_file_as)
        open_history_action = action('&Open', self.open_history)
        quit_action = action('&Quit', self.close)


        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(open_action)
        fileMenu.addAction(open_dir_action)
        fileMenu.addAction(quit_action)

        historyMenu = menubar.addMenu('&History')
        historyMenu.addAction(open_history_action)
        historyMenu.addAction(save_action)
        historyMenu.addAction(save_as_action)

        configMenu = menubar.addMenu('&Config')
        config_model_action = action('&Model Config', self.model_config)
        configMenu.addAction(config_model_action)

        self.about_us_menu = menubar.addMenu('&About us')
        self.about_us_menu.aboutToShow.connect(self.AboutUsDialog.show)

        self.worker = TestThread()
        self.worker.progressSignal.connect(self.progress_slot)
        self.worker.errorMsgSignal.connect(self.error_msg_slot)
        self.worker.msgSignal.connect(self.status)
        self.worker.testFinishedSignal.connect(self.result_show)
        self.worker.modelLoaded.connect(self.model_loaded_flag)
        self.worker.start()


    def init_dialogs(self):
        self.ConfigDialog = config_widget.ModelConfig()
        self.ConfigDialog.save_button.clicked.connect(self.load_model_trigger)
        self.AboutUsDialog = AboutUs()
        

    def file_search_changed(self):
        pattern = self.file_search.text()
        self.show_search(pattern=pattern)

    def file_select_changed(self):
        pass

    def open_single_file(self):
        pass

    def open_directory(self):
        test_path = os.path.split(cfg.TestPath)[0]
        path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Open Dir', test_path)
        if path:
            cfg.TestPath = path
            cfg.update_settings()
            self.path_edit.setText(path)

    def open_history(self):
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'History File', './history')
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.results = data['results']
            self.file_list_widget.clear()
            for item in self.results:
                self.file_list_widget.addItem(item['filename'])
            # show first line
            if len(self.results):
                self.file_list_widget.setCurrentRow(0)

    def save_file(self):
        pass

    def save_file_as(self):
        pass

    def model_config(self):
        self.ConfigDialog.show()

    def model_loaded_flag(self):
        self.load_button.setStyleSheet('background-color: rgb(85, 255, 255);')

    def load_model_trigger(self):
        style = self.start_button.styleSheet()
        self.load_button.setStyleSheet(style)
        angle_md = cfg.AngleModel if self.ConfigDialog.angle_ckbox.isChecked() else None
        detect_md = cfg.DetectModel if self.ConfigDialog.detect_ckbox.isChecked() else None
        recog_md = cfg.RecognizeModel if self.ConfigDialog.recog_ckbox.isChecked() else None
        print(angle_md)
        print(detect_md)
        print(recog_md)
        self.worker.load_model(angle_md=angle_md,
                               detect_md=detect_md,
                               recog_md=recog_md)

    def run(self):
        self.start_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.worker.add_task(self.path_edit.text())

    def result_show(self):
        self.results = copy.deepcopy(self.worker.results)
        save_to_json('test',
                     results = self.results,
                     angle_model=cfg.AngleModel,
                     detect_model=cfg.DetectModel,
                     recog_model=cfg.RecognizeModel,
                     lexicon_file=cfg.LexiconFile)
        self.file_list_widget.clear()
        for item in self.results:
            self.file_list_widget.addItem(item['filename'])
        # show first line
        if len(self.results):
            self.file_list_widget.setCurrentRow(0)
        self.start_button.setEnabled(True)
        self.open_button.setEnabled(True)
        self.load_button.setEnabled(True)

    def show_search(self, pattern=''):
        self.file_list_widget.clear()
        for item in self.results:
            if pattern in item['filename']:
                self.file_list_widget.addItem(item['filename'])

    def file_select_changed(self):
        items = self.file_list_widget.selectedItems()
        if not items:
            return
        item = items[0]
        filename = str(item.text())
        result_item = [it for it in self.results if it['filename']==filename][0]

        labels = []

        # load image
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        # img = contrast_one(img)
        if result_item.get('angle'):
            angle = int(result_item['angle'])
            if angle == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif angle == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif angle == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if result_item.get('angle_adjust'):
            angle = float(result_item['angle_adjust'])
            rows,cols = img.shape[0], img.shape[1]
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.warpAffine(img,M,(cols,rows))
        if result_item.get('rect'):
            x, y, w, h = result_item['rect']
            img = img[y:y+h, x:x+w, :]

        # draw boxes and label
        for label_item in result_item['labels']:
            box = list(label_item['box'])
            text = label_item['label']
            labels.append(text)
            fontScale = (box[3]-box[1]) / 64
            # cv2.rectangle(img, box[:2], box[2:], (0, 0, 255), 2)
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            # cv2.putText(img, text, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 255, 0), 2)
            # img = paint_chinese_opencv(img, text, (box[0], box[1]), int(32), (0, 255, 0))

        # show image
        rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QtGui.QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
        
        pixmap = QtGui.QPixmap.fromImage(convertToQtFormat)
        scaledPixmap = pixmap.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaledPixmap)

        # zp: 20121211
        if result_item.get('label_show'):
            labels = list(result_item['label_show'])

        # show label info
        self.label_widget.clear()
        self.label_widget.addItems(labels)

    def progress_slot(self, count):
        self.progress_bar.setValue(count)

    def error_msg_slot(self, msg):
        QtWidgets.QMessageBox.warning(self, 'Error', str(msg))

    def status(self, msg, delay=5000):
        self.statusBar().showMessage(msg, delay)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    # win.showMaximized()
    win.resize(1280, 900)
    win.show()
    sys.exit(app.exec_())