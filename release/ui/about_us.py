import os
import sys

from PyQt5 import QtCore
from PyQt5 import QtGui
from PyQt5 import QtWidgets


class AboutUs(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('About us')
        self.text = "    上海夏数科技是一家致力于经营大数据科学与深度学习的网络科技公司，" + \
                    "也是同济大学科技园人工智能科技创新企业，公司核心研发人员均毕业于世界前" + \
                    "20名高校。\n\n地址：上海市杨浦区国康路46号同济科技大厦417室\n电话：" + \
                    "17721027662\n邮箱：liguangze@xiashutech.com"
        self.label = QtWidgets.QLabel()
        self.label.setText(self.text)
        self.label.setStyleSheet('font: 10pt "微软雅黑";')
        self.label.setWordWrap(True)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.resize(500, 300)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = AboutUs()
    win.resize(500,300)
    win.show()
    sys.exit(app.exec_())