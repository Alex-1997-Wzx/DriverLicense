import os

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets

import ui.config as cfg


class ModelConfig(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Model Config')
        self.setWindowIcon(QtGui.QIcon('./ui/icons/model.png'))

        angle_lb = QtWidgets.QLabel('Angle Model')
        self.angle_edit = QtWidgets.QLineEdit()
        self.angle_edit.setPlaceholderText('选填')
        self.angle_button = QtWidgets.QPushButton('...')
        self.angle_button.clicked.connect(self.select_angel_model)
        detect_lb = QtWidgets.QLabel('Detect Model')
        self.detect_edit = QtWidgets.QLineEdit()
        self.detect_edit.setPlaceholderText('必填')
        self.detect_button = QtWidgets.QPushButton('...')
        self.detect_button.clicked.connect(self.select_detect_model)
        recog_lb = QtWidgets.QLabel('Recognize Model')
        self.recog_edit = QtWidgets.QLineEdit()
        self.recog_edit.setPlaceholderText('必填')
        self.recog_button = QtWidgets.QPushButton('...')
        self.recog_button.clicked.connect(self.select_recognize_model)
        lexicon_lb = QtWidgets.QLabel('Lexicon File')
        self.lexicon_edit =QtWidgets.QLineEdit()
        self.lexicon_edit.setPlaceholderText('选填')
        self.lexcion_button = QtWidgets.QPushButton('...')
        self.lexcion_button.clicked.connect(self.select_lexicon_file)

        # setenable
        self.angle_edit.setEnabled(False)
        self.lexicon_edit.setEnabled(False)

        # checkbox
        self.angle_ckbox = QtWidgets.QCheckBox()
        self.detect_ckbox = QtWidgets.QCheckBox()
        self.detect_ckbox.setCheckState(QtCore.Qt.Checked)
        self.recog_ckbox = QtWidgets.QCheckBox()
        self.recog_ckbox.setCheckState(QtCore.Qt.Checked)
        self.lexicon_ckbox = QtWidgets.QCheckBox()
        for ckbox in [self.angle_ckbox, self.detect_ckbox, self.recog_ckbox, self.lexicon_ckbox]:
            # ckbox.setCheckState(QtCore.Qt.Checked)
            ckbox.stateChanged.connect(self.checkbox_clicked)

        gridLayout = QtWidgets.QGridLayout()
        gridLayout.addWidget(angle_lb, 0, 0)
        gridLayout.addWidget(self.angle_edit, 0, 1)
        gridLayout.addWidget(self.angle_button, 0, 2)
        gridLayout.addWidget(self.angle_ckbox, 0, 3)
        gridLayout.addWidget(detect_lb, 1, 0)
        gridLayout.addWidget(self.detect_edit, 1, 1)
        gridLayout.addWidget(self.detect_button, 1, 2)
        gridLayout.addWidget(self.detect_ckbox, 1, 3)
        gridLayout.addWidget(recog_lb, 2, 0)
        gridLayout.addWidget(self.recog_edit, 2, 1)
        gridLayout.addWidget(self.recog_button, 2, 2)
        gridLayout.addWidget(self.recog_ckbox, 2, 3)
        gridLayout.addWidget(lexicon_lb, 3, 0)
        gridLayout.addWidget(self.lexicon_edit, 3, 1)
        gridLayout.addWidget(self.lexcion_button, 3, 2)
        gridLayout.addWidget(self.lexicon_ckbox, 3, 3)

        self.close_button = QtWidgets.QPushButton('Close')
        self.close_button.clicked.connect(self.close)
        self.save_button = QtWidgets.QPushButton('Load')
        self.save_button.clicked.connect(self.close)
        hlayout = QtWidgets.QHBoxLayout()
        h_spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        hlayout.addSpacerItem(h_spacer)
        hlayout.addWidget(self.close_button)
        hlayout.addWidget(self.save_button)

        vlayout = QtWidgets.QVBoxLayout()
        vlayout.addLayout(gridLayout)
        vlayout.addLayout(hlayout)
        self.setLayout(vlayout)

        self.initialize()
        self.resize(800, 300)

    def initialize(self):
        if os.path.exists(cfg.AngleModel):
            self.angle_edit.setText(cfg.AngleModel)
        if os.path.exists(cfg.DetectModel):
            self.detect_edit.setText(cfg.DetectModel)
        if os.path.exists(cfg.RecognizeModel):
            self.recog_edit.setText(cfg.RecognizeModel)
        if os.path.exists(cfg.LexiconFile):
            self.lexicon_edit.setText(cfg.LexiconFile)

    def select_angel_model(self):
        angle_path = os.path.split(cfg.AngleModel)[0]
        # path = QtWidgets.QFileDialog.getExistingDirectory(self, 'Angle Model', angle_path)
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Angle Model', angle_path)
        if filename:
            cfg.AngleModel = filename
            cfg.update_settings()
            self.angle_edit.setText(filename)

    def select_detect_model(self):
        detect_path = os.path.split(cfg.DetectModel)[0]
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Detect Model', detect_path)
        if filename:
            cfg.DetectModel = filename
            cfg.update_settings()
            self.detect_edit.setText(filename)

    def select_recognize_model(self):
        recog_path = os.path.split(cfg.RecognizeModel)[0]
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Recognize Model', recog_path)
        if filename:
            cfg.RecognizeModel = filename
            cfg.update_settings()
            self.recog_edit.setText(filename)

    def select_lexicon_file(self):
        lexicon_path = os.path.split(cfg.LexiconFile)[0]
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Lexicon File', lexicon_path)
        if filename:
            cfg.LexiconFile = filename
            cfg.update_settings()
            self.lexicon_edit.setText(filename)

    def checkbox_clicked(self, state):
        sender = self.sender()
        if sender == self.angle_ckbox:
            self.angle_edit.setEnabled(state==QtCore.Qt.Checked)
        elif sender == self.detect_ckbox:
            self.detect_edit.setEnabled(state=QtCore.Qt.Checked)
        elif sender == self.recog_ckbox:
            self.recog_edit.setEnabled(state==QtCore.Qt.Checked)
        elif sender == self.lexicon_ckbox:
            self.lexicon_edit.setEnabled(state==QtCore.Qt.Checked)