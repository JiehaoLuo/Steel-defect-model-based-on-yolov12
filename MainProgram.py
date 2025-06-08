# -*- coding: utf-8 -*-
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
                             QWidget, QHeaderView, QTableWidgetItem, QAbstractItemView,
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QLabel,
                             QLineEdit, QTableWidget, QComboBox, QFrame, QSplitter,
                             QGroupBox, QProgressBar as QProgressBarWidget)
import sys
import os
from PIL import ImageFont
from ultralytics import YOLO

sys.path.append('UIProgram')
# 保留原有的import接口，但不使用
try:
    from UIProgram.UiMain import Ui_MainWindow
    from UIProgram.QssLoader import QSSLoader
    from UIProgram.precess_bar import ProgressBar
except:
    print("原有UI模块导入失败，使用新的现代化界面")

from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QCoreApplication, QPropertyAnimation, QEasingCurve, QRect
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor, QLinearGradient
import detect_tools as tools
import cv2
import Config
import numpy as np
import torch


class ModernButton(QPushButton):
    def __init__(self, text, color="#4CAF50", compact=False):
        super().__init__(text)

        if compact:
            # 紧凑型按钮 - 更像正方形
            self.setFixedSize(80, 35)
            font_size = "12px"
            padding = "5px"
        else:
            # 标准按钮
            self.setFixedHeight(35)
            self.setMaximumWidth(120)
            font_size = "13px"
            padding = "8px 15px"

        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {color}, stop:1 {self.darken_color(color)});
                color: white;
                border: none;
                border-radius: 6px;
                font-size: {font_size};
                font-weight: bold;
                padding: {padding};
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                    stop:0 {self.lighten_color(color)}, stop:1 {color});
                transform: translateY(-1px);
            }}
            QPushButton:pressed {{
                background: {self.darken_color(color)};
                transform: translateY(0px);
            }}
        """)

    def darken_color(self, color):
        # 简单的颜色加深
        color_map = {
            "#4CAF50": "#388E3C",
            "#2196F3": "#1976D2",
            "#FF9800": "#F57C00",
            "#9C27B0": "#7B1FA2",
            "#F44336": "#D32F2F"
        }
        return color_map.get(color, "#333333")

    def lighten_color(self, color):
        # 简单的颜色变亮
        color_map = {
            "#4CAF50": "#66BB6A",
            "#2196F3": "#42A5F5",
            "#FF9800": "#FFB74D",
            "#9C27B0": "#BA68C8",
            "#F44336": "#EF5350"
        }
        return color_map.get(color, "#555555")


class ModernCard(QFrame):
    def __init__(self, title=""):
        super().__init__()
        self.setStyleSheet("""
            QFrame {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 12px;
                margin: 5px;
            }
        """)
        self.setContentsMargins(10, 10, 10, 10)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        if title:
            title_label = QLabel(title)
            title_label.setStyleSheet("""
                QLabel {
                    color: #FFFFFF;
                    font-size: 16px;
                    font-weight: bold;
                    margin-bottom: 6px;
                    padding: 3px 0px;
                    text-align: center;
                    qproperty-alignment: AlignCenter;
                }
            """)
            title_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_label)


class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(QMainWindow, self).__init__(parent)

        # 设置窗口属性
        self.setWindowTitle("基于YOLOV12的钢材表面缺陷识别检测系统")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        # 设置深色主题
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1E1E1E;
                color: #FFFFFF;
            }
        """)

        # 初始化变量
        self.conf = 0.3
        self.iou = 0.7
        self.show_width = 770
        self.show_height = 480
        self.org_path = None
        self.is_camera_open = False
        self.cap = None

        # 创建UI
        self.setupUI()
        self.initMain()
        self.signalconnect()

    def setupUI(self):
        """创建现代化的用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 左侧面板
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # 右侧面板
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # 设置分割比例
        splitter.setSizes([1000, 350])
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #404040;
                width: 3px;
            }
            QSplitter::handle:hover {
                background-color: #4CAF50;
            }
        """)

    def create_left_panel(self):
        """创建左侧面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # 图像显示区域
        image_card = ModernCard("图像显示区域")
        image_layout = QVBoxLayout()

        self.label_show = QLabel()
        self.label_show.setMinimumSize(770, 480)
        self.label_show.setAlignment(Qt.AlignCenter)
        self.label_show.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2D2D2D, stop:1 #3D3D3D);
                border: 2px dashed #4CAF50;
                border-radius: 8px;
                color: #CCCCCC;
                font-size: 16px;
                padding: 20px;
            }
        """)
        self.label_show.setText("图像显示区域\n请选择图片或视频文件")

        image_layout.addWidget(self.label_show)
        image_card.layout().addLayout(image_layout)
        left_layout.addWidget(image_card)

        # 检测结果表格
        table_card = ModernCard("检测结果与位置信息")
        table_layout = QVBoxLayout()

        self.tableWidget = QTableWidget()
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setHorizontalHeaderLabels(['序号', '文件路径', '类别', '置信度', '坐标位置'])
        self.tableWidget.setStyleSheet("""
            QTableWidget {
                background-color: #2D2D2D;
                border: 1px solid #404040;
                border-radius: 8px;
                gridline-color: #404040;
                color: #FFFFFF;
                font-size: 13px;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #404040;
            }
            QTableWidget::item:selected {
                background-color: #4CAF50;
                color: white;
            }
            QHeaderView::section {
                background-color: #404040;
                color: #FFFFFF;
                padding: 12px;
                border: none;
                font-weight: bold;
                font-size: 14px;
            }
        """)

        table_layout.addWidget(self.tableWidget)
        table_card.layout().addLayout(table_layout)
        left_layout.addWidget(table_card)

        return left_widget

    def create_right_panel(self):
        """创建右侧控制面板"""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(8)

        # 文件导入区域
        file_card = ModernCard("文件导入")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(6)

        # 图片文件 - 使用网格布局让按钮更紧凑
        pic_grid = QGridLayout()
        pic_grid.setSpacing(8)

        self.PicBtn = ModernButton("📷", "#4CAF50", compact=True)
        self.FilesBtn = ModernButton("📁", "#2196F3", compact=True)

        pic_label = QLabel("图片:")
        pic_label.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: 500;")
        pic_grid.addWidget(pic_label, 0, 0)
        pic_grid.addWidget(self.PicBtn, 0, 1)
        pic_grid.addWidget(self.FilesBtn, 0, 2)
        pic_grid.setColumnStretch(3, 1)  # 让剩余空间拉伸

        file_layout.addLayout(pic_grid)

        self.PiclineEdit = QLineEdit()
        self.PiclineEdit.setPlaceholderText("图片路径将显示在这里")
        self.PiclineEdit.setStyleSheet(self.get_lineedit_style())
        file_layout.addWidget(self.PiclineEdit)

        # 视频文件
        video_grid = QGridLayout()
        video_grid.setSpacing(8)

        self.VideoBtn = ModernButton("🎥", "#FF9800", compact=True)

        video_label = QLabel("视频:")
        video_label.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: 500;")
        video_grid.addWidget(video_label, 0, 0)
        video_grid.addWidget(self.VideoBtn, 0, 1)
        video_grid.setColumnStretch(2, 1)

        file_layout.addLayout(video_grid)

        self.VideolineEdit = QLineEdit()
        self.VideolineEdit.setPlaceholderText("视频路径将显示在这里")
        self.VideolineEdit.setStyleSheet(self.get_lineedit_style())
        file_layout.addWidget(self.VideolineEdit)

        # 摄像头
        camera_grid = QGridLayout()
        camera_grid.setSpacing(8)

        self.CapBtn = ModernButton("📹", "#9C27B0", compact=True)

        camera_label = QLabel("摄像头:")
        camera_label.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: 500;")
        camera_grid.addWidget(camera_label, 0, 0)
        camera_grid.addWidget(self.CapBtn, 0, 1)
        camera_grid.setColumnStretch(2, 1)

        file_layout.addLayout(camera_grid)

        self.CaplineEdit = QLineEdit()
        self.CaplineEdit.setText("摄像头未开启")
        self.CaplineEdit.setReadOnly(True)
        self.CaplineEdit.setStyleSheet(self.get_lineedit_style())
        file_layout.addWidget(self.CaplineEdit)

        file_card.layout().addLayout(file_layout)
        right_layout.addWidget(file_card)

        # 检测结果区域
        result_card = ModernCard("检测结果")
        result_layout = QGridLayout()
        result_layout.setSpacing(6)

        # 创建信息标签样式
        label_style = """
            QLabel {
                color: #CCCCCC;
                font-size: 15px;
                padding: 6px;
                font-weight: 500;
                text-align: center;
                qproperty-alignment: AlignCenter;
                background-color: #404040;
                border-radius: 4px;
            }
        """

        value_style = """
            QLabel {
                color: #4CAF50;
                font-size: 16px;
                font-weight: bold;
                padding: 6px;
                background-color: #333333;
                border-radius: 4px;
                min-width: 60px;
                text-align: center;
                qproperty-alignment: AlignCenter;
            }
        """

        # 第一行：用时和目标数量
        time_label = QLabel("用时")
        time_label.setStyleSheet(label_style)
        time_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(time_label, 0, 0)

        self.time_lb = QLabel("0.000 s")
        self.time_lb.setStyleSheet(value_style)
        self.time_lb.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.time_lb, 0, 1)

        nums_label = QLabel("数量")
        nums_label.setStyleSheet(label_style)
        nums_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(nums_label, 0, 2)

        self.label_nums = QLabel("0")
        self.label_nums.setStyleSheet(value_style)
        self.label_nums.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.label_nums, 0, 3)

        # 第二行：目标选择
        target_label = QLabel("目标")
        target_label.setStyleSheet(label_style)
        target_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(target_label, 1, 0)

        self.comboBox = QComboBox()
        self.comboBox.addItem("全部")
        self.comboBox.setStyleSheet("""
            QComboBox {
                background-color: #333333;
                color: #FFFFFF;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 6px;
                font-size: 14px;
                min-height: 20px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #333333;
                color: #FFFFFF;
                selection-background-color: #4CAF50;
                font-size: 14px;
            }
        """)
        result_layout.addWidget(self.comboBox, 1, 1, 1, 3)

        # 第三行：类型和置信度
        type_label = QLabel("类型")
        type_label.setStyleSheet(label_style)
        type_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(type_label, 2, 0)

        self.type_lb = QLabel("无")
        self.type_lb.setStyleSheet(value_style)
        self.type_lb.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.type_lb, 2, 1)

        conf_label = QLabel("置信度")
        conf_label.setStyleSheet(label_style)
        conf_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(conf_label, 2, 2)

        self.label_conf = QLabel("0.00%")
        self.label_conf.setStyleSheet(value_style)
        self.label_conf.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.label_conf, 2, 3)

        # 目标位置标题
        pos_label = QLabel("目标位置")
        pos_label.setStyleSheet(label_style + "font-weight: bold; margin-top: 8px;")
        pos_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(pos_label, 3, 0, 1, 4)

        # 坐标信息
        coord_style = """
            QLabel {
                color: #FFD700;
                font-size: 15px;
                font-weight: bold;
                padding: 5px;
                background-color: #333333;
                border-radius: 4px;
                min-width: 50px;
                text-align: center;
                qproperty-alignment: AlignCenter;
            }
        """

        # 坐标标签样式
        coord_label_style = """
            QLabel {
                color: #CCCCCC;
                font-size: 15px;
                font-weight: 600;
                padding: 5px;
                text-align: center;
                qproperty-alignment: AlignCenter;
                background-color: #404040;
                border-radius: 4px;
            }
        """

        # X坐标
        xmin_label = QLabel("xmin")
        xmin_label.setStyleSheet(coord_label_style)
        xmin_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(xmin_label, 4, 0)

        self.label_xmin = QLabel("0")
        self.label_xmin.setStyleSheet(coord_style)
        self.label_xmin.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.label_xmin, 4, 1)

        xmax_label = QLabel("xmax")
        xmax_label.setStyleSheet(coord_label_style)
        xmax_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(xmax_label, 4, 2)

        self.label_xmax = QLabel("0")
        self.label_xmax.setStyleSheet(coord_style)
        self.label_xmax.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.label_xmax, 4, 3)

        # Y坐标
        ymin_label = QLabel("ymin")
        ymin_label.setStyleSheet(coord_label_style)
        ymin_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(ymin_label, 5, 0)

        self.label_ymin = QLabel("0")
        self.label_ymin.setStyleSheet(coord_style)
        self.label_ymin.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.label_ymin, 5, 1)

        ymax_label = QLabel("ymax")
        ymax_label.setStyleSheet(coord_label_style)
        ymax_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(ymax_label, 5, 2)

        self.label_ymax = QLabel("0")
        self.label_ymax.setStyleSheet(coord_style)
        self.label_ymax.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.label_ymax, 5, 3)

        result_card.layout().addLayout(result_layout)
        right_layout.addWidget(result_card)

        # 操作区域
        action_card = ModernCard("操作")
        action_layout = QHBoxLayout()
        action_layout.setSpacing(8)

        self.SaveBtn = ModernButton("💾 保存", "#F44336", compact=False)
        self.ExitBtn = ModernButton("🚪 退出", "#607D8B", compact=False)

        operation_label = QLabel("操作:")
        operation_label.setStyleSheet("color: #FFFFFF; font-size: 15px; font-weight: 500;")
        action_layout.addWidget(operation_label)
        action_layout.addWidget(self.SaveBtn)
        action_layout.addWidget(self.ExitBtn)
        action_layout.addStretch()

        action_card.layout().addLayout(action_layout)
        right_layout.addWidget(action_card)

        # 添加弹性空间
        right_layout.addStretch()

        return right_widget

    def get_lineedit_style(self):
        return """
            QLineEdit {
                background-color: #333333;
                color: #FFFFFF;
                border: 1px solid #404040;
                border-radius: 4px;
                padding: 8px;
                font-size: 13px;
                max-height: 25px;
            }
            QLineEdit:focus {
                border: 2px solid #4CAF50;
            }
        """

    def signalconnect(self):
        """连接信号槽"""
        self.PicBtn.clicked.connect(self.open_img)
        self.comboBox.activated.connect(self.combox_change)
        self.VideoBtn.clicked.connect(self.vedio_show)
        self.CapBtn.clicked.connect(self.camera_show)
        self.SaveBtn.clicked.connect(self.save_detect_video)
        self.ExitBtn.clicked.connect(QCoreApplication.quit)
        self.FilesBtn.clicked.connect(self.detact_batch_imgs)

    def initMain(self):
        """初始化主要组件"""
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")

        # 加载检测模型
        try:
            self.model = YOLO(Config.model_path, task='detect')
            self.model(np.zeros((48, 48, 3)), device=self.device)
            print("✓ 模型加载成功")
        except Exception as e:
            print(f"✗ 模型加载失败: {e}")

        # 字体加载
        try:
            self.fontC = ImageFont.truetype("Font/platech.ttf", 25, 0)
            print("✓ 字体加载成功")
        except Exception as e:
            print(f"✗ 字体加载失败，使用默认字体: {e}")

        # 用于绘制不同颜色矩形框
        self.colors = tools.Colors()

        # 定时器
        self.timer_camera = QTimer()
        self.timer_save_video = QTimer()

        # 表格设置
        self.setup_table()

    def setup_table(self):
        """设置表格"""
        table = self.tableWidget
        table.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        table.verticalHeader().setDefaultSectionSize(40)
        table.setColumnWidth(0, 80)
        table.setColumnWidth(1, 250)
        table.setColumnWidth(2, 120)
        table.setColumnWidth(3, 100)
        table.setColumnWidth(4, 250)
        table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)
        # 让表格水平填充
        table.horizontalHeader().setStretchLastSection(True)

    # === 功能方法 ===

    def open_img(self):
        """打开图片文件"""
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        file_path, _ = QFileDialog.getOpenFileName(None, '打开图片', './', "Image files (*.jpg *.jpeg *.png *.bmp)")
        if not file_path:
            return

        self.comboBox.setDisabled(False)
        self.org_path = file_path
        self.org_img = tools.img_cvread(self.org_path)

        # 目标检测
        t1 = time.time()
        self.results = self.model(self.org_path, conf=self.conf, iou=self.iou)[0]
        t2 = time.time()
        take_time_str = '{:.3f} s'.format(t2 - t1)
        self.time_lb.setText(take_time_str)

        location_list = self.results.boxes.xyxy.tolist()
        self.location_list = [list(map(int, e)) for e in location_list]
        cls_list = self.results.boxes.cls.tolist()
        self.cls_list = [int(i) for i in cls_list]
        self.conf_list = self.results.boxes.conf.tolist()
        self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

        now_img = self.results.plot()
        self.draw_img = now_img
        self.img_width, self.img_height = self.get_resize_size(now_img)
        resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)
        self.PiclineEdit.setText(self.org_path)

        target_nums = len(self.cls_list)
        self.label_nums.setText(str(target_nums))

        choose_list = ['全部']
        target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
        choose_list = choose_list + target_names

        self.comboBox.clear()
        self.comboBox.addItems(choose_list)

        if target_nums >= 1:
            self.type_lb.setText(Config.CH_names[self.cls_list[0]])
            self.label_conf.setText(str(self.conf_list[0]))
            self.label_xmin.setText(str(self.location_list[0][0]))
            self.label_ymin.setText(str(self.location_list[0][1]))
            self.label_xmax.setText(str(self.location_list[0][2]))
            self.label_ymax.setText(str(self.location_list[0][3]))
        else:
            self.type_lb.setText('无')
            self.label_conf.setText('0.00%')
            self.label_xmin.setText('0')
            self.label_ymin.setText('0')
            self.label_xmax.setText('0')
            self.label_ymax.setText('0')

        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=self.org_path)

    def combox_change(self):
        """下拉框改变"""
        if not hasattr(self, 'results'):
            return

        com_text = self.comboBox.currentText()
        if com_text == '全部':
            cur_box = self.location_list
            cur_img = self.results.plot()
            if self.cls_list:
                self.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.label_conf.setText(str(self.conf_list[0]))
        else:
            index = int(com_text.split('_')[-1])
            cur_box = [self.location_list[index]]
            cur_img = self.results[index].plot()
            self.type_lb.setText(Config.CH_names[self.cls_list[index]])
            self.label_conf.setText(str(self.conf_list[index]))

        if cur_box:
            self.label_xmin.setText(str(cur_box[0][0]))
            self.label_ymin.setText(str(cur_box[0][1]))
            self.label_xmax.setText(str(cur_box[0][2]))
            self.label_ymax.setText(str(cur_box[0][3]))

        resize_cvimg = cv2.resize(cur_img, (self.img_width, self.img_height))
        pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
        self.label_show.clear()
        self.label_show.setPixmap(pix_img)
        self.label_show.setAlignment(Qt.AlignCenter)

    def vedio_show(self):
        """显示视频"""
        if self.is_camera_open:
            self.is_camera_open = False
            self.CaplineEdit.setText('摄像头未开启')

        video_path = self.get_video_path()
        if not video_path:
            return None
        self.cap = cv2.VideoCapture(video_path)
        self.video_start()
        self.comboBox.setDisabled(True)

    def camera_show(self):
        """摄像头功能"""
        self.is_camera_open = not self.is_camera_open
        if self.is_camera_open:
            self.CaplineEdit.setText('摄像头开启')
            self.cap = cv2.VideoCapture(0)
            self.video_start()
            self.comboBox.setDisabled(True)
        else:
            self.CaplineEdit.setText('摄像头未开启')
            if self.cap:
                self.cap.release()
                cv2.destroyAllWindows()
            self.label_show.clear()
            self.label_show.setText("图像显示区域\n请选择图片或视频文件")

    def save_detect_video(self):
        """保存检测结果"""
        if not hasattr(self, 'org_path') or not self.org_path:
            QMessageBox.information(self, '提示', '当前没有可保存信息，请先打开图片或视频！')
            return

        if self.is_camera_open:
            QMessageBox.information(self, '提示', '摄像头视频无法保存!')
            return

        if hasattr(self, 'draw_img'):
            fileName = os.path.basename(self.org_path)
            name, end_name = fileName.rsplit(".", 1)
            save_name = name + '_detect_result.' + end_name
            save_img_path = os.path.join(Config.save_path, save_name)
            cv2.imwrite(save_img_path, self.draw_img)
            QMessageBox.information(self, '保存成功', f'图片保存成功!\n文件路径: {save_img_path}')
        else:
            QMessageBox.information(self, '提示', '没有检测结果可保存！')

    def detact_batch_imgs(self):
        """批量检测图片"""
        if self.cap:
            self.video_stop()
            self.is_camera_open = False
            self.CaplineEdit.setText('摄像头未开启')
            self.cap = None

        directory = QFileDialog.getExistingDirectory(self, "选取文件夹", "./")
        if not directory:
            return

        self.org_path = directory
        img_suffix = ['jpg', 'png', 'jpeg', 'bmp']

        for file_name in os.listdir(directory):
            full_path = os.path.join(directory, file_name)
            if os.path.isfile(full_path) and file_name.split('.')[-1].lower() in img_suffix:
                img_path = full_path
                self.org_img = tools.img_cvread(img_path)

                t1 = time.time()
                self.results = self.model(img_path, conf=self.conf, iou=self.iou)[0]
                t2 = time.time()
                take_time_str = '{:.3f} s'.format(t2 - t1)
                self.time_lb.setText(take_time_str)

                location_list = self.results.boxes.xyxy.tolist()
                self.location_list = [list(map(int, e)) for e in location_list]
                cls_list = self.results.boxes.cls.tolist()
                self.cls_list = [int(i) for i in cls_list]
                self.conf_list = self.results.boxes.conf.tolist()
                self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

                now_img = self.results.plot()
                self.draw_img = now_img

                self.img_width, self.img_height = self.get_resize_size(now_img)
                resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
                pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
                self.label_show.setPixmap(pix_img)
                self.label_show.setAlignment(Qt.AlignCenter)
                self.PiclineEdit.setText(img_path)

                target_nums = len(self.cls_list)
                self.label_nums.setText(str(target_nums))

                choose_list = ['全部']
                target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
                choose_list = choose_list + target_names

                self.comboBox.clear()
                self.comboBox.addItems(choose_list)

                if target_nums >= 1:
                    self.type_lb.setText(Config.CH_names[self.cls_list[0]])
                    self.label_conf.setText(str(self.conf_list[0]))
                    self.label_xmin.setText(str(self.location_list[0][0]))
                    self.label_ymin.setText(str(self.location_list[0][1]))
                    self.label_xmax.setText(str(self.location_list[0][2]))
                    self.label_ymax.setText(str(self.location_list[0][3]))

                self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path=img_path)
                self.tableWidget.scrollToBottom()
                QApplication.processEvents()

    # === 辅助方法 ===

    def get_video_path(self):
        file_path, _ = QFileDialog.getOpenFileName(None, '打开视频', './', "Video files (*.avi *.mp4 *.wmv *.mkv)")
        if not file_path:
            return None
        self.org_path = file_path
        self.VideolineEdit.setText(file_path)
        return file_path

    def video_start(self):
        self.tableWidget.setRowCount(0)
        self.tableWidget.clearContents()
        self.comboBox.clear()
        self.timer_camera.start(30)
        self.timer_camera.timeout.connect(self.open_frame)

    def video_stop(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        self.timer_camera.stop()

    def open_frame(self):
        ret, now_img = self.cap.read()
        if ret:
            t1 = time.time()
            results = self.model(now_img, conf=self.conf, iou=self.iou)[0]
            t2 = time.time()
            take_time_str = '{:.3f} s'.format(t2 - t1)
            self.time_lb.setText(take_time_str)

            location_list = results.boxes.xyxy.tolist()
            self.location_list = [list(map(int, e)) for e in location_list]
            cls_list = results.boxes.cls.tolist()
            self.cls_list = [int(i) for i in cls_list]
            self.conf_list = results.boxes.conf.tolist()
            self.conf_list = ['%.2f %%' % (each * 100) for each in self.conf_list]

            now_img = results.plot()
            self.img_width, self.img_height = self.get_resize_size(now_img)
            resize_cvimg = cv2.resize(now_img, (self.img_width, self.img_height))
            pix_img = tools.cvimg_to_qpiximg(resize_cvimg)
            self.label_show.setPixmap(pix_img)
            self.label_show.setAlignment(Qt.AlignCenter)

            target_nums = len(self.cls_list)
            self.label_nums.setText(str(target_nums))

            choose_list = ['全部']
            target_names = [Config.names[id] + '_' + str(index) for index, id in enumerate(self.cls_list)]
            choose_list = choose_list + target_names

            self.comboBox.clear()
            self.comboBox.addItems(choose_list)

            if target_nums >= 1:
                self.type_lb.setText(Config.CH_names[self.cls_list[0]])
                self.label_conf.setText(str(self.conf_list[0]))
                self.label_xmin.setText(str(self.location_list[0][0]))
                self.label_ymin.setText(str(self.location_list[0][1]))
                self.label_xmax.setText(str(self.location_list[0][2]))
                self.label_ymax.setText(str(self.location_list[0][3]))

            self.tabel_info_show(self.location_list, self.cls_list, self.conf_list, path="实时检测")
        else:
            self.video_stop()

    def get_resize_size(self, img):
        _img = img.copy()
        img_height, img_width, depth = _img.shape
        ratio = img_width / img_height
        if ratio >= self.show_width / self.show_height:
            self.img_width = self.show_width
            self.img_height = int(self.img_width / ratio)
        else:
            self.img_height = self.show_height
            self.img_width = int(self.img_height * ratio)
        return self.img_width, self.img_height

    def tabel_info_show(self, locations, clses, confs, path=None):
        for location, cls, conf in zip(locations, clses, confs):
            row_count = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_count)

            items = [
                QTableWidgetItem(str(row_count + 1)),
                QTableWidgetItem(str(path) if path else ""),
                QTableWidgetItem(str(Config.CH_names[cls])),
                QTableWidgetItem(str(conf)),
                QTableWidgetItem(str(location))
            ]

            for i, item in enumerate(items):
                if i in [0, 2, 3]:
                    item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                self.tableWidget.setItem(row_count, i, item)

        self.tableWidget.scrollToBottom()


if __name__ == "__main__":
    print("=== 现代化检测系统启动 ===")

    # 高DPI设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)

    app = QApplication(sys.argv)

    # 设置应用程序字体
    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    try:
        win = MainWindow()
        win.show()

        print("=== 界面已显示 ===")
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback

        traceback.print_exc()