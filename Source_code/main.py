# main.py

import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytesseract
from PIL import Image, ImageTk
import csv
import threading
import utlis  # Đảm bảo rằng utlis.py nằm trong cùng thư mục
import smtplib
import re
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from symspellpy import SymSpell, Verbosity
from difflib import get_close_matches

def preprocess_text(text):
    # Làm sạch văn bản OCR
    text = re.sub(r"[^a-zA-ZĐđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ0-9,._@&()\-\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_tesseract_path():
    if sys.platform.startswith('win'):
        possible_paths = [
            r'C:/Program Files/Tesseract-OCR/tesseract.exe'
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

class BusinessCardExtractor:
    def __init__(self, model_path, tesseract_path, dictionary_path='TuDienv2.txt'):
        self.model = YOLO(model_path)
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.config = '--oem 3 --psm 6 -l vie'
        self.dictionary = self.load_dictionary(dictionary_path)
        self.initialize_symspell(dictionary_path)

    def load_dictionary(self, dictionary_path):
        """
        Tải từ điển từ file .txt và lưu vào thuộc tính của lớp.
        """
        try:
            with open(dictionary_path, 'r', encoding='utf-8') as f:
                words = set(f.read().splitlines())
            print(f"Tải từ điển thành công! Số lượng từ: {len(words)}")
            return words
        except Exception as e:
            print(f"Lỗi khi tải từ điển: {e}")
            return set()

    def initialize_symspell(self, dictionary_path):
        try:
            self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            self.symspell.load_dictionary(dictionary_path, term_index=0, count_index=1, encoding='utf-8')
        except:
            self.symspell = None

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binary

    def detect_objects(self, image_path):
        results = self.model(image_path)
        return results[0]

    def crop_objects(self, image, results):
        cropped_objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            class_id = int(box.cls)
            cropped = image[int(y1):int(y2), int(x1):int(x2)]
            cropped_objects.append({'image': cropped, 'class': class_id})
        return cropped_objects

    def extract_text(self, image):
        processed_image = self.preprocess_image(image)
        raw_text = pytesseract.image_to_string(processed_image, config=self.config)
        cleaned_text = preprocess_text(raw_text)
        if self.symspell:
            corrected_text = self.correct_text_with_symspell(cleaned_text)
        else:
            corrected_text = cleaned_text
        return corrected_text.strip()

    def correct_text_with_symspell(self, text):
        if not self.symspell:
            return text
        corrected_words = []
        for word in text.split():
            suggestions = self.symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected_words.append(suggestions[0].term)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

    def process_business_card(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Không thể đọc ảnh")
        results = self.detect_objects(image_path)
        cropped_objects = self.crop_objects(image, results)

        extracted_info = {
            0: [], # Tên
            1: [], # Số điện thoại
            2: [], # Email
            3: [], # Địa chỉ
            4: []  # Chức vụ
        }

        for obj in cropped_objects:
            text = self.extract_text(obj['image'])
            if obj['class'] in extracted_info:
                extracted_info[obj['class']].append(text)
        return extracted_info

class BusinessCardExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trích xuất thông tin danh thiếp")
        self.root.geometry("1200x800")

        self.original_image = None
        self.max_image_width = 800
        self.max_image_height = 400
        self.batch_processing = False
        self.processing_thread = None
        self.batch_results = []

        self.available_models = {
            'YOLOv8 Nano (Lightweight)': './yolov8n_.pt',
            'YOLOv9': './yolov9t.pt',
            'YOLOv10 Nano': './yolov10.pt',
            'YOLOv11 Nano': './yolov11n.pt'
        }

        self.setup_tesseract()

        if hasattr(self, 'tesseract_path'):
            self.threshold1 = 200
            self.threshold2 = 150
            self.notebook = ttk.Notebook(self.root)
            self.notebook.pack(fill="both", expand=True)

            # Tabs
            self.tab1 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab1, text="Xử lý Ảnh")

            self.tab2 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab2, text="Chụp Live")

            self.tab3 = ttk.Frame(self.notebook)
            self.notebook.add(self.tab3, text="Gửi Email")

            self.extractor = BusinessCardExtractor(
                model_path=self.available_models['YOLOv8 Nano (Lightweight)'],
                tesseract_path=self.tesseract_path,
                dictionary_path="./TuDienv2.txt"
            )

            self.setup_gui_tab1()
            self.setup_gui_tab2()
            self.setup_gui_tab3()

    def setup_tesseract(self):
        self.tesseract_path = get_tesseract_path()
        if not self.tesseract_path:
            response = messagebox.askquestion("Cấu hình Tesseract",
                "Không tìm thấy Tesseract OCR. Cấu hình thủ công?")
            if response == 'yes':
                self.tesseract_path = filedialog.askopenfilename(
                    title="Chọn file tesseract.exe",
                    filetypes=[("Executable files", "*.exe"), ("All files", "*.*")]
                )
                if not self.tesseract_path:
                    messagebox.showerror("Lỗi", "Không thể tiếp tục mà không có Tesseract OCR.")
                    self.root.destroy()
            else:
                messagebox.showerror("Lỗi", "Không thể tiếp tục mà không có Tesseract OCR.")
                self.root.destroy()

    def setup_gui_tab1(self):
        main_frame = ttk.Frame(self.tab1, padding="10")
        main_frame.pack(fill="both", expand=True)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=5)

        model_selection_frame = ttk.Frame(control_frame)
        model_selection_frame.pack(fill="x", pady=5)

        tk.Label(model_selection_frame, text="Chọn mô hình:").grid(row=0, column=0, padx=(0,10), sticky='w')
        self.model_var = tk.StringVar(self.root)
        self.model_var.set('YOLOv8 Nano (Lightweight)')
        self.model_dropdown = ttk.OptionMenu(
            model_selection_frame,
            self.model_var,
            'YOLOv8 Nano (Lightweight)',
            *self.available_models.keys(),
            "Custom Model",
            command=self.on_model_select
        )
        self.model_dropdown.grid(row=0, column=1, sticky='w')

        self.custom_model_label = ttk.Label(self.tab1, text="Đường dẫn mô hình tùy chỉnh:")
        self.custom_model_entry = ttk.Entry(self.tab1, width=50)
        self.browse_model_button = ttk.Button(self.tab1, text="Duyệt", command=self.browse_custom_model)

        # Buttons
        self.select_btn = ttk.Button(control_frame, text="Chọn ảnh", command=self.select_image)
        self.select_btn.pack(side="left", padx=5)

        self.process_btn = ttk.Button(control_frame, text="Xử lý", command=self.process_image)
        self.process_btn.pack(side="left", padx=5)
        self.process_btn['state'] = 'disabled'

        self.select_folder_btn = ttk.Button(control_frame, text="Chọn thư mục ảnh", command=self.select_folder)
        self.select_folder_btn.pack(side="left", padx=5)

        self.process_folder_btn = ttk.Button(control_frame, text="Xử lý thư mục", command=self.process_folder)
        self.process_folder_btn.pack(side="left", padx=5)
        self.process_folder_btn['state'] = 'disabled'

        self.save_btn = ttk.Button(control_frame, text="Lưu vào CSV", command=self.save_to_existing_csv)
        self.save_btn.pack(side="left", padx=5)
        self.save_btn['state'] = 'disabled'

        # Tạo nút Lưu tất cả 1 lần
        self.save_all_btn = ttk.Button(control_frame, text="Lưu tất cả", command=self.save_batch_results)
        self.save_all_btn.pack(side="left", padx=5)
        self.save_all_btn['state'] = 'disabled'

        self.load_csv_btn = ttk.Button(control_frame, text="Load dữ liệu từ CSV", command=self.load_data_from_csv)
        self.load_csv_btn.pack(side="left", padx=5)

        self.view_extracted_btn = ttk.Button(control_frame, text="Xem dữ liệu đã trích xuất", command=self.view_extracted_data)
        self.view_extracted_btn.pack(side="left", padx=5)
        self.view_extracted_btn['state'] = 'disabled'

        # Progress
        self.progress_frame = ttk.Frame(main_frame)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)
        self.progress_label = ttk.Label(self.progress_frame, text="")

        self.path_label = ttk.Label(main_frame, text="Chưa chọn ảnh")
        self.path_label.pack(fill="x", pady=5)

        self.current_processing_label = ttk.Label(main_frame, text="", foreground="blue")
        self.current_processing_label.pack(fill="x", pady=5)

        self.image_frame = ttk.LabelFrame(main_frame, text="Ảnh danh thiếp")
        self.image_frame.pack(fill="both", expand=True, padx=(5,0), pady=5)

        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(pady=5, expand=True)

        results_frame = ttk.LabelFrame(main_frame, text="Kết quả trích xuất")
        results_frame.pack(fill="both", expand=True, pady=5)

        self.result_tree = ttk.Treeview(results_frame, columns=("Tên", "Số điện thoại", "Email", "Địa chỉ", "Chức vụ"), show="headings")
        self.result_tree.heading("Tên", text="Tên")
        self.result_tree.heading("Số điện thoại", text="Số điện thoại")
        self.result_tree.heading("Email", text="Email")
        self.result_tree.heading("Địa chỉ", text="Địa chỉ")
        self.result_tree.heading("Chức vụ", text="Chức vụ")

        self.result_tree.column("Tên", width=200)
        self.result_tree.column("Số điện thoại", width=150)
        self.result_tree.column("Email", width=300)
        self.result_tree.column("Địa chỉ", width=300)
        self.result_tree.column("Chức vụ", width=200)

        tree_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=tree_scrollbar.set)
        self.result_tree.pack(side="left", fill="both", expand=True)
        tree_scrollbar.pack(side="right", fill="y")

        self.result_tree.bind("<MouseWheel>", self._on_mousewheel_treeview)

    def setup_gui_tab2(self):
        # Tab Chụp Live
        main_frame = ttk.Frame(self.tab2, padding="10")
        main_frame.pack(fill="both", expand=True)

        # Frame nhập URL
        url_control_frame = ttk.Frame(main_frame)
        url_control_frame.pack(fill="x", pady=5)

        ttk.Label(url_control_frame, text="DroidCam URL:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.droidcam_url_var = tk.StringVar(value="http://192.168.0.100:4747/video")
        self.droidcam_url_entry = ttk.Entry(url_control_frame, textvariable=self.droidcam_url_var, width=40)
        self.droidcam_url_entry.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        update_url_btn = ttk.Button(url_control_frame, text="Cập nhật URL", command=self.update_droidcam_url)
        update_url_btn.grid(row=0, column=2, padx=5, pady=5, sticky="w")


        # Điều khiển video
        update_url_btn = ttk.Button(url_control_frame, text="Cập nhật URL", command=self.update_droidcam_url)
        update_url_btn.grid(row=0, column=2, padx=5, pady=5, sticky="w")

        # Các nút điều khiển video đặt ngang hàng với URL
        self.start_video_btn = ttk.Button(url_control_frame, text="Bắt đầu Video", command=self.start_video)
        self.start_video_btn.grid(row=0, column=3, padx=5, pady=5, sticky="w")

        self.stop_video_btn = ttk.Button(url_control_frame, text="Dừng Video", command=self.stop_video, state='disabled')
        self.stop_video_btn.grid(row=0, column=4, padx=5, pady=5, sticky="w")

        self.capture_btn = ttk.Button(url_control_frame, text="Chụp Ảnh", command=self.capture_image, state='disabled')
        self.capture_btn.grid(row=0, column=5, padx=5, pady=5, sticky="w")

        # Frame để chứa slider chỉnh Threshold
        slider_frame = ttk.LabelFrame(main_frame, text="Chỉnh Threshold")
        slider_frame.pack(fill="x", pady=5)

        # Slider Threshold1
        ttk.Label(slider_frame, text="Threshold 1:").pack(side="left", padx=5)
        self.threshold1_var = tk.IntVar(value=self.threshold1)
        self.threshold1_slider = ttk.Scale(slider_frame, from_=0, to=255, variable=self.threshold1_var, orient="horizontal", command=self.update_threshold)
        self.threshold1_slider.pack(side="left", fill="x", expand=True, padx=5)

        # Slider Threshold2
        ttk.Label(slider_frame, text="Threshold 2:").pack(side="left", padx=5)
        self.threshold2_var = tk.IntVar(value=self.threshold2)
        self.threshold2_slider = ttk.Scale(slider_frame, from_=0, to=255, variable=self.threshold2_var, orient="horizontal", command=self.update_threshold)
        self.threshold2_slider.pack(side="left", fill="x", expand=True, padx=5)


        # Video hiển thị với kích thước nhỏ hơn
        video_frame = ttk.LabelFrame(main_frame, text="Live Video")
        video_frame.pack(pady=5)

        self.video_label = ttk.Label(video_frame)
        self.video_label.pack()
        self.video_label.config()  # Đặt kích thước nhỏ hơn


        self.cap = None
        self.video_running = False
        self.captured_image = None


    def setup_gui_tab3(self):
        # Tab Gửi Email
        email_frame = ttk.Frame(self.tab3, padding="10")
        email_frame.pack(fill="both", expand=True)

        # Combobox lọc Chức vụ
        filter_frame = ttk.LabelFrame(email_frame, text="Bộ lọc")
        filter_frame.pack(fill="x", pady=10)

        ttk.Label(filter_frame, text="Chức vụ:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.position_filter = tk.StringVar()
        self.position_combobox = ttk.Combobox(filter_frame, textvariable=self.position_filter, state="readonly")
        self.position_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.position_combobox.bind("<<ComboboxSelected>>", self.filter_email_list)

        ttk.Label(filter_frame, text="Địa chỉ:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.address_filter = tk.StringVar()
        self.address_combobox = ttk.Combobox(filter_frame, textvariable=self.address_filter, state="readonly")
        self.address_combobox.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.address_combobox.bind("<<ComboboxSelected>>", self.filter_email_list)        

        ttk.Label(email_frame, text="Tiêu đề email:").pack(anchor="w", pady=5)
        self.email_subject = tk.Entry(email_frame, width=50)
        self.email_subject.pack(fill="x", pady=5)

        ttk.Label(email_frame, text="Nội dung email:").pack(anchor="w", pady=5)

        self.email_body = tk.Text(email_frame, height=15, wrap="word")
        self.email_body.insert("1.0", "Chào {name},\n\nChúng tôi rất vui mừng được giới thiệu đến bạn sản phẩm mới của chúng tôi...\n")
        self.email_body.pack(fill="both", pady=5)

        def clear_placeholder(event):
            current_content = self.email_body.get("1.0", tk.END).strip()
            placeholder_text = "Chào {name},\n\nChúng tôi rất vui mừng được giới thiệu đến bạn sản phẩm mới của chúng tôi..."
            if current_content == placeholder_text:
                self.email_body.delete("1.0", tk.END)

        def restore_placeholder(event):
            current_content = self.email_body.get("1.0", tk.END).strip()
            placeholder_text = "Chào {name},\n\nChúng tôi rất vui mừng được giới thiệu đến bạn sản phẩm mới của chúng tôi..."
            if not current_content:
                self.email_body.insert("1.0", placeholder_text)

        self.email_body.bind("<FocusIn>", clear_placeholder)
        self.email_body.bind("<FocusOut>", restore_placeholder)

        load_frame = ttk.Frame(email_frame)
        load_frame.pack(fill="x", pady=5)
        self.load_csv_btn_email = ttk.Button(load_frame, text="Load CSV", command=self.load_email_csv)
        self.load_csv_btn_email.pack(side="left", padx=5)

        send_button = ttk.Button(load_frame, text="Gửi Email", command=self.send_emails)
        send_button.pack(side="left", padx=5)

        self.email_list_frame = ttk.Frame(email_frame)
        self.email_list_frame.pack(fill="both", expand=True, pady=(5,0))

        columns = ("Tên", "Email")
        self.email_tree = ttk.Treeview(self.email_list_frame, columns=columns, show="headings")
        self.email_tree.heading("Tên", text="Tên")
        self.email_tree.heading("Email", text="Email")
        self.email_tree.column("Tên", width=200)
        self.email_tree.column("Email", width=300)

        email_tree_scrollbar = ttk.Scrollbar(self.email_list_frame, orient="vertical", command=self.email_tree.yview)
        self.email_tree.configure(yscrollcommand=email_tree_scrollbar.set)
        email_tree_scrollbar.pack(side="right", fill="y")
        self.email_tree.pack(fill="both", expand=True)
        

        edit_frame = ttk.Frame(email_frame)
        edit_frame.pack(fill="x", pady=5)

        ttk.Label(edit_frame, text="Tên:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        self.new_name_entry = ttk.Entry(edit_frame)
        self.new_name_entry.grid(row=0, column=1, padx=5, pady=5, sticky='w')

        ttk.Label(edit_frame, text="Email:").grid(row=0, column=2, padx=5, pady=5, sticky='e')
        self.new_email_entry = ttk.Entry(edit_frame)
        self.new_email_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')

        # Các nút trên cùng một hàng
        self.add_email_btn = ttk.Button(edit_frame, text="Thêm email", command=self.add_new_email)
        self.add_email_btn.grid(row=0, column=4, padx=5, pady=5, sticky='w')

        self.delete_email_btn = ttk.Button(edit_frame, text="Xóa email đã chọn", command=self.delete_selected_email)
        self.delete_email_btn.grid(row=0, column=5, padx=5, pady=5, sticky='w')

        self.edit_email_btn = ttk.Button(edit_frame, text="Sửa email đã chọn", command=self.edit_selected_email)
        self.edit_email_btn.grid(row=0, column=6, padx=5, pady=5, sticky='w')

    def on_model_select(self, selection):
        if selection == 'Custom Model':
            self.custom_model_label.grid(row=3, column=0, pady=(10, 0), sticky='w')
            self.custom_model_entry.grid(row=3, column=1, pady=(0, 10), sticky='w')
            self.browse_model_button.grid(row=3, column=2, pady=(0, 10), sticky='w')
        else:
            self.custom_model_label.grid_remove()
            self.custom_model_entry.grid_remove()
            self.browse_model_button.grid_remove()
        try:
            model_path = (
                self.custom_model_entry.get()
                if selection == 'Custom Model'
                else self.available_models[selection]
            )
            self.extractor = BusinessCardExtractor(model_path=model_path, tesseract_path=self.tesseract_path)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")

    def browse_custom_model(self):
        filename = filedialog.askopenfilename(
            title="Chọn tệp mô hình",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.custom_model_entry.delete(0, tk.END)
            self.custom_model_entry.insert(0, filename)
            try:
                self.extractor = BusinessCardExtractor(model_path=filename, tesseract_path=self.tesseract_path)
            except Exception as e:
                messagebox.showerror("Lỗi", f"Không thể tải mô hình: {str(e)}")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self.path_label.config(text=file_path)
            self.process_btn['state'] = 'normal'
            self.show_image(file_path)

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Chọn thư mục chứa ảnh danh thiếp")
        if folder_path:
            self.current_folder_path = folder_path
            self.image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))
            ]
            if not self.image_files:
                messagebox.showwarning("Cảnh báo", "Không tìm thấy file ảnh nào trong thư mục này!")
                return
            self.current_file_index = 0
            self.batch_results = []

            first_image_path = os.path.join(folder_path, self.image_files[0])
            self.show_image(first_image_path)

            self.path_label.config(
                text=f"Thư mục: {folder_path}\nẢnh {self.image_files[self.current_file_index]}"
            )
            self.process_folder_btn['state'] = 'normal'
            self.process_btn['state'] = 'disabled'

    def process_folder(self):
        if not hasattr(self, 'current_folder_path'):
            return
        self.progress_frame.pack(fill="x", pady=5)
        self.progress_bar.pack(fill="x", padx=5)
        self.progress_label.pack(pady=2)
        self.progress_var.set(0)
        self.batch_processing = True
        self.disable_buttons()

        self.processing_thread = threading.Thread(target=self._process_folder_thread, daemon=True)
        self.processing_thread.start()
        self.root.after(100, self.check_batch_progress)

    def _process_folder_thread(self):
        try:
            total_images = len(self.image_files)
            self.batch_results = []

            for idx, image_file in enumerate(self.image_files):
                image_path = os.path.join(self.current_folder_path, image_file)
                self.root.after(0, self.update_current_image, image_path, idx)
                result = self.process_single_image(image_path)
                if result:
                    self.batch_results.append(result)
                self.progress_var.set(((idx + 1) / total_images) * 100)

            self.root.after(0, self.show_batch_results, 0)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý thư mục: {str(e)}")
        finally:
            self.batch_processing = False

    def update_current_image(self, image_path, index):
        self.current_file_index = index
        self.show_image(image_path)
        self.current_processing_label.config(text=f"Đang xử lý ảnh {index + 1}/{len(self.image_files)}: {self.image_files[index]}")
        self.path_label.config(
            text=f"Thư mục: {self.current_folder_path}\nẢnh {self.image_files[index]}"
        )

    def show_batch_results(self, index=None):
        if not self.batch_results:
            return
        if index is None:
            index = self.current_file_index

        for item in self.result_tree.get_children():
            self.result_tree.delete(item)

        result = self.batch_results[index]
        self.result_tree.insert("", "end", values=(
            ', '.join(result['results'][0]),
            ', '.join(result['results'][1]),
            ', '.join(result['results'][2]),
            ', '.join(result['results'][3]),
            ', '.join(result['results'][4])
        ))

        self.enable_navigation_buttons()
        self.view_extracted_btn['state'] = 'normal'

    def enable_navigation_buttons(self):
        if not hasattr(self, 'nav_frame'):
            self.nav_frame = ttk.Frame(self.tab1)
            self.nav_frame.pack(fill="x", pady=5)

            self.prev_btn = ttk.Button(self.nav_frame, text="←", command=self.show_previous)
            self.prev_btn.pack(side="left", padx=5)

            self.next_btn = ttk.Button(self.nav_frame, text="→", command=self.show_next)
            self.next_btn.pack(side="left", padx=5)

        if self.batch_results and len(self.batch_results) > 0:
            self.save_all_btn['state'] = 'normal'
        else:
            self.save_all_btn['state'] = 'disabled'

        self.update_navigation_state()

    def show_previous(self):
        if self.current_file_index > 0:
            self.current_file_index -= 1
            image_path = os.path.join(self.current_folder_path, self.image_files[self.current_file_index])
            self.show_image(image_path)
            self.show_batch_results(self.current_file_index)
            self.update_navigation_state()

    def show_next(self):
        if self.current_file_index < len(self.batch_results) - 1:
            self.current_file_index += 1
            image_path = os.path.join(self.current_folder_path, self.image_files[self.current_file_index])
            self.show_image(image_path)
            self.show_batch_results(self.current_file_index)
            self.update_navigation_state()

    def update_navigation_state(self):
        self.prev_btn['state'] = 'normal' if self.current_file_index > 0 else 'disabled'
        self.next_btn['state'] = 'normal' if self.current_file_index < len(self.batch_results) - 1 else 'disabled'
        if hasattr(self, 'current_folder_path'):
            self.path_label.config(
                text=f"Thư mục: {self.current_folder_path}\nẢnh {self.image_files[self.current_file_index]}"
            )

    def process_single_image(self, image_path):
        try:
            results = self.extractor.process_business_card(image_path)
            return {'file': os.path.basename(image_path), 'results': results}
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def save_batch_results(self):
        if not self.batch_results:
            return
        try:
            csv_path = "./data_export.csv"
            with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                for result in self.batch_results:
                    if result and 'results' in result:
                        writer.writerow([
                            ', '.join(result['results'][0]),
                            ', '.join(result['results'][1]),
                            ', '.join(result['results'][2]),
                            ', '.join(result['results'][3]),
                            ', '.join(result['results'][4])
                        ])
            messagebox.showinfo("Thành công", f"Đã lưu kết quả của {len(self.batch_results)} ảnh vào {csv_path}!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu vào CSV: {str(e)}")

    def check_batch_progress(self):
        if self.batch_processing:
            progress = self.progress_var.get()
            self.progress_label.config(text=f"Đang xử lý... {progress:.1f}%")
            self.root.after(100, self.check_batch_progress)
        else:
            self.progress_label.config(text="Hoàn thành!")
            self.enable_buttons()
            self.progress_frame.pack_forget()

    def disable_buttons(self):
        self.select_btn['state'] = 'disabled'
        self.select_folder_btn['state'] = 'disabled'
        self.process_btn['state'] = 'disabled'
        self.process_folder_btn['state'] = 'disabled'
        self.save_btn['state'] = 'disabled'
        self.load_csv_btn['state'] = 'disabled'
        self.view_extracted_btn['state'] = 'disabled'
        self.save_all_btn['state'] = 'disabled'

    def enable_buttons(self):
        self.select_btn['state'] = 'normal'
        self.select_folder_btn['state'] = 'normal'
        if hasattr(self, 'current_image_path'):
            self.process_btn['state'] = 'normal'
        if hasattr(self, 'current_folder_path'):
            self.process_folder_btn['state'] = 'normal'
        self.load_csv_btn['state'] = 'normal'

    def show_image(self, image_path):
        self.original_image = Image.open(image_path)
        display_image = self.resize_image(self.original_image)
        photo = ImageTk.PhotoImage(display_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def process_image(self):
        try:
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)

            results = self.extractor.process_business_card(self.current_image_path)
            self.extracted_results = results

            self.result_tree.insert("", "end", values=(
                ', '.join(results[0]),
                ', '.join(results[1]),
                ', '.join(results[2]),
                ', '.join(results[3]),
                ', '.join(results[4])
            ))
            self.save_btn['state'] = 'normal'
        except Exception as e:
            messagebox.showerror("Lỗi", f"Lỗi khi xử lý ảnh: {str(e)}")

    def save_to_existing_csv(self):
        existing_csv_path = "./data_export.csv"
        try:
            with open(existing_csv_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([
                    ', '.join(self.extracted_results[0]),
                    ', '.join(self.extracted_results[1]),
                    ', '.join(self.extracted_results[2]),
                    ', '.join(self.extracted_results[3]),
                    ', '.join(self.extracted_results[4])
                ])
            messagebox.showinfo("Thành công", f"Lưu kết quả vào {existing_csv_path} thành công!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể lưu vào CSV: {str(e)}")

    def load_data_from_csv(self):
        csv_file = filedialog.askopenfilename(
            title="Chọn file CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not csv_file:
            return
        try:
            data = pd.read_csv(csv_file)
            required_columns = {"Tên", "Số điện thoại", "Email", "Địa chỉ", "Chức vụ"}
            if not required_columns.issubset(data.columns):
                raise ValueError("File CSV không chứa các cột yêu cầu.")

            for item in self.result_tree.get_children():
                self.result_tree.delete(item)

            for _, row in data.iterrows():
                self.result_tree.insert("", "end", values=(
                    row["Tên"], row["Số điện thoại"], row["Email"], row["Địa chỉ"], row["Chức vụ"]
                ))

            # Có thể trực quan hóa dữ liệu theo chức vụ
            self.visualize_roles(data)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load dữ liệu: {str(e)}")

    def visualize_roles(self, data):
        role_counts = data["Chức vụ"].value_counts()
        plt.figure(figsize=(15, 8))
        role_counts.plot(kind="barh", color="skyblue")
        plt.title("Số lượng theo chức vụ")
        plt.ylabel("Chức vụ")
        plt.xlabel("Số lượng")
        plt.tight_layout()
        plt.show()

    def update_droidcam_url(self):
        """
        Cập nhật URL DroidCam và dừng video đang chạy nếu có.
        """
        if self.video_running:
            self.stop_video()

        self.droidcam_url = self.droidcam_url_var.get()
        messagebox.showinfo("Cập nhật URL", f"URL mới đã được cập nhật: {self.droidcam_url}")


    def start_video(self):
        if not self.video_running:
            self.droidcam_url = self.droidcam_url_var.get()  # Lấy URL mới từ Entry
            self.cap = cv2.VideoCapture(self.droidcam_url)
            if not self.cap.isOpened():
                messagebox.showerror("Lỗi", f"Không thể kết nối với DroidCam tại {self.droidcam_url}")
                return
            self.video_running = True
            self.start_video_btn['state'] = 'disabled'
            self.stop_video_btn['state'] = 'normal'
            self.capture_btn['state'] = 'normal'
            self.video_loop()


    def stop_video(self):
        if self.video_running:
            self.video_running = False
            if self.cap:
                self.cap.release()
            self.video_label.config(image='')
            self.start_video_btn['state'] = 'normal'
            self.stop_video_btn['state'] = 'disabled'
            self.capture_btn['state'] = 'disabled'

    def video_loop(self):
        if not self.video_running:
            return

        try:
            success, frame = self.cap.read()
            if not success:
                messagebox.showerror("Lỗi", "Không thể đọc khung hình từ DroidCam.")
                self.stop_video()
                return

            # 1. Thu nhỏ kích thước video gốc
            frame = cv2.resize(frame, (320, 240))

            # 2. Chuyển sang Gray và Threshold
            imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 0.55)
            imgThreshold = cv2.Canny(imgBlur, self.threshold1, self.threshold2)

            # 3. Tìm và vẽ Contours
            contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            imgContours = frame.copy()
            cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 2)

            # 4. Phát hiện và Warp Perspective
            biggest, _ = utlis.biggestContour(contours)
            if biggest.size != 0:
                biggest = utlis.reorder(biggest)
                pts1 = np.float32(biggest)
                pts2 = np.float32([[0, 0], [320, 0], [0, 240], [320, 240]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                imgWarpColored = cv2.warpPerspective(frame, matrix, (320, 240))
            else:
                imgWarpColored = frame.copy()

            # 5. Nối 4 video để hiển thị
            top_row = np.hstack((frame, cv2.cvtColor(imgThreshold, cv2.COLOR_GRAY2BGR)))  # Video gốc và threshold
            bottom_row = np.hstack((imgContours, imgWarpColored))  # Video contours và kết quả cuối cùng
            stacked = np.vstack((top_row, bottom_row))  # Nối 2 hàng lại

            # 6. Hiển thị video
            stacked = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(stacked)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

            self.root.after(10, self.video_loop)

        except Exception as e:
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")
            self.stop_video()



    def capture_image(self):
        if self.video_running:
            try:
                success, frame = self.cap.read()
                if not success:
                    messagebox.showerror("Lỗi", "Không thể chụp ảnh.")
                    return
                #càng cao càng rõ
                frame = cv2.resize(frame, (640, 640))
                imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                imgBlur = cv2.GaussianBlur(imgGray, (3,3), 0.55)
                thres = (self.threshold1, self.threshold2)
                imgThreshold = cv2.Canny(imgBlur, thres[0], thres[1])
                kernel = np.ones((5,5))
                imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
                imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

                contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                biggest, maxArea = utlis.biggestContour(contours)
                if biggest.size != 0:
                    biggest = utlis.reorder(biggest)
                    pts1 = np.float32(biggest)
                    pts2 = np.float32([[0,0],[480,0],[0,640],[480,640]])
                    matrix = cv2.getPerspectiveTransform(pts1, pts2)
                    imgWarpColored = cv2.warpPerspective(frame, matrix, (480,640))

                    imgWarpColored = imgWarpColored[5:imgWarpColored.shape[0]-5,5:imgWarpColored.shape[1]-5]
                    imgWarpColored = cv2.resize(imgWarpColored, (640,360))

                    save_path = f"Scanned/myImage{self.get_next_count()}.jpg"
                    os.makedirs("Scanned", exist_ok=True)
                    cv2.imwrite(save_path, imgWarpColored)

                    messagebox.showinfo("Thành công", f"Ảnh đã được lưu tại {save_path}")
            except Exception as e:
                messagebox.showerror("Lỗi", f"Lỗi khi chụp ảnh: {str(e)}")

    def get_next_count(self):
        if not os.path.exists("Scanned"):
            return 0
        existing_files = os.listdir("Scanned")
        counts = [int(f.replace("myImage","").replace(".jpg","")) for f in existing_files if f.startswith("myImage") and f.endswith(".jpg")]
        return max(counts)+1 if counts else 0

    def view_extracted_data(self):
        if not self.batch_results:
            messagebox.showwarning("Cảnh báo", "Chưa có dữ liệu để xem.")
            return

        view_window = tk.Toplevel(self.root)
        view_window.title("Xem dữ liệu đã trích xuất")
        view_window.geometry("1000x700")

        view_notebook = ttk.Notebook(view_window)
        view_notebook.pack(fill="both", expand=True)

        images_tab = ttk.Frame(view_notebook)
        view_notebook.add(images_tab, text="Hình ảnh")

        data_tab = ttk.Frame(view_notebook)
        view_notebook.add(data_tab, text="Dữ liệu")

        images_canvas = tk.Canvas(images_tab)
        scrollbar = ttk.Scrollbar(images_tab, orient="vertical", command=images_canvas.yview)
        scrollable_frame = ttk.Frame(images_canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: images_canvas.configure(
                scrollregion=images_canvas.bbox("all")
            )
        )

        images_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        images_canvas.configure(yscrollcommand=scrollbar.set)

        images_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for result in self.batch_results:
            frame = ttk.Frame(scrollable_frame, padding=10)
            frame.pack(fill="x", pady=5)

            image_path = os.path.join(self.current_folder_path, result['file'])
            try:
                pil_image = Image.open(image_path)
                pil_image = self.resize_image(pil_image)
                img = ImageTk.PhotoImage(pil_image)
                img_label = ttk.Label(frame, image=img)
                img_label.image = img
                img_label.pack(side="left")
            except:
                img_label = ttk.Label(frame, text="Không thể mở ảnh")
                img_label.pack(side="left")

            data_text = (
                f"Tên: {', '.join(result['results'][0])}\n"
                f"Số điện thoại: {', '.join(result['results'][1])}\n"
                f"Email: {', '.join(result['results'][2])}\n"
                f"Địa chỉ: {', '.join(result['results'][3])}\n"
                f"Chức vụ: {', '.join(result['results'][4])}"
            )
            data_label = ttk.Label(frame, text=data_text, justify="left")
            data_label.pack(side="left", padx=10)

        data_tree = ttk.Treeview(data_tab, columns=("Tên", "Số điện thoại", "Email", "Địa chỉ", "Chức vụ"), show="headings")
        data_tree.heading("Tên", text="Tên")
        data_tree.heading("Số điện thoại", text="Số điện thoại")
        data_tree.heading("Email", text="Email")
        data_tree.heading("Địa chỉ", text="Địa chỉ")
        data_tree.heading("Chức vụ", text="Chức vụ")

        data_tree.column("Tên", width=150)
        data_tree.column("Số điện thoại", width=120)
        data_tree.column("Email", width=200)
        data_tree.column("Địa chỉ", width=250)
        data_tree.column("Chức vụ", width=150)

        data_scrollbar = ttk.Scrollbar(data_tab, orient="vertical", command=data_tree.yview)
        data_tree.configure(yscrollcommand=data_scrollbar.set)
        data_tree.pack(side="left", fill="both", expand=True)
        data_scrollbar.pack(side="right", fill="y")

        for result in self.batch_results:
            data_tree.insert("", "end", values=(
                ', '.join(result['results'][0]),
                ', '.join(result['results'][1]),
                ', '.join(result['results'][2]),
                ', '.join(result['results'][3]),
                ', '.join(result['results'][4])
            ))

    def update_threshold(self, event=None):
        self.threshold1 = int(self.threshold1_var.get())
        self.threshold2 = int(self.threshold2_var.get())

    def update_email_tree(self, data):
        """
        Cập nhật dữ liệu vào Treeview email.
        """
        # Xóa tất cả dữ liệu cũ trong Treeview
        for i in self.email_tree.get_children():
            self.email_tree.delete(i)

        # Thêm dữ liệu mới từ DataFrame vào Treeview
        for _, row in data.iterrows():
            self.email_tree.insert("", "end", values=(
                row["Tên"], 
                row["Email"], 
                row["Chức vụ"], 
                row["Địa chỉ"]
            ))



    def add_new_email(self):
        name = self.new_name_entry.get().strip()
        email = self.new_email_entry.get().strip()
        if not name or not email:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ Tên và Email.")
            return
        self.email_tree.insert("", "end", values=(name, email))
        self.new_name_entry.delete(0, tk.END)
        self.new_email_entry.delete(0, tk.END)

    def delete_selected_email(self):
        selected_item = self.email_tree.selection()
        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một dòng để xóa.")
            return
        for item in selected_item:
            self.email_tree.delete(item)

    def edit_selected_email(self):
        selected_item = self.email_tree.selection()
        if not selected_item:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn một dòng để sửa.")
            return
        if len(selected_item) > 1:
            messagebox.showwarning("Cảnh báo", "Vui lòng chỉ chọn một dòng để sửa.")
            return

        item = selected_item[0]
        name, email = self.email_tree.item(item, "values")

        edit_dialog = tk.Toplevel(self.root)
        edit_dialog.title("Sửa Email")

        ttk.Label(edit_dialog, text="Tên:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
        name_entry = ttk.Entry(edit_dialog)
        name_entry.insert(0, name)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(edit_dialog, text="Email:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
        email_entry = ttk.Entry(edit_dialog)
        email_entry.insert(0, email)
        email_entry.grid(row=1, column=1, padx=5, pady=5)

        def save_changes():
            new_name = name_entry.get().strip()
            new_email = email_entry.get().strip()
            if not new_name or not new_email:
                messagebox.showwarning("Cảnh báo", "Vui lòng nhập đầy đủ thông tin.")
                return
            self.email_tree.item(item, values=(new_name, new_email))
            edit_dialog.destroy()

        save_button = ttk.Button(edit_dialog, text="Lưu", command=save_changes)
        save_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

    def load_email_csv(self):
        csv_file = filedialog.askopenfilename(
            title="Chọn file CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not csv_file:
            return

        try:
            self.email_data = pd.read_csv(csv_file, encoding="utf-8")
            if not {"Email", "Tên", "Chức vụ", "Địa chỉ"}.issubset(self.email_data.columns):
                messagebox.showerror("Lỗi", "File CSV phải chứa các cột: 'Tên', 'Email', 'Chức vụ', 'Địa chỉ'.")
                return

            # Lọc danh sách các giá trị duy nhất
            positions = sorted(self.email_data["Chức vụ"].dropna().unique().tolist())
            addresses = sorted(self.email_data["Địa chỉ"].dropna().unique().tolist())

            self.position_combobox['values'] = ["Tất cả"] + positions
            self.address_combobox['values'] = ["Tất cả"] + addresses
            self.position_combobox.set("Tất cả")
            self.address_combobox.set("Tất cả")

            self.update_email_tree(self.email_data)

            messagebox.showinfo("Thành công", f"Đã tải dữ liệu từ file CSV!")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể load file CSV: {str(e)}")


    def filter_email_list(self, event=None):
        filtered_data = self.email_data.copy()

        if self.position_filter.get() != "Tất cả":
            filtered_data = filtered_data[filtered_data["Chức vụ"] == self.position_filter.get()]

        if self.address_filter.get() != "Tất cả":
            filtered_data = filtered_data[filtered_data["Địa chỉ"] == self.address_filter.get()]

        self.update_email_tree(filtered_data)


    def send_emails(self):
        subject = self.email_subject.get().strip()
        body_template = self.email_body.get("1.0", tk.END).strip()

        if not subject or not body_template:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập tiêu đề và nội dung email.")
            return

        email_list = [
            (self.email_tree.item(item, "values")[0], self.email_tree.item(item, "values")[1])
            for item in self.email_tree.get_children()
        ]

        if not email_list:
            messagebox.showwarning("Cảnh báo", "Không có email nào trong danh sách sau khi lọc.")
            return

        # Gửi email logic (như cũ)
        SMTP_SERVER = 'smtp.gmail.com'
        SMTP_PORT = 587
        EMAIL_ADDRESS = 'nptruong60@gmail.com'  # Thay bằng email của bạn
        EMAIL_PASSWORD = 'qjun pdhp bvnv waxm'  # Thay bằng mật khẩu ứng dụng

        success_count = 0
        failure_count = 0

        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể kết nối đến SMTP server: {e}")
            return

        for recipient_name, recipient_email in email_list:
            try:
                personalized_body = body_template.replace("{name}", recipient_name)
                msg = MIMEMultipart()
                msg['From'] = EMAIL_ADDRESS
                msg['To'] = recipient_email
                msg['Subject'] = subject
                msg.attach(MIMEText(personalized_body, 'plain'))
                server.sendmail(EMAIL_ADDRESS, recipient_email, msg.as_string())
                success_count += 1
            except Exception as e:
                print(f"Lỗi khi gửi email tới {recipient_email}: {e}")
                failure_count += 1

        server.quit()
        messagebox.showinfo("Kết quả", f"Gửi email thành công: {success_count}\nGửi email thất bại: {failure_count}")
    
    def _on_mousewheel_treeview(self, event):
        if sys.platform.startswith('win'):
            delta = -1 * (event.delta // 120)
        elif sys.platform == 'darwin':
            delta = -1 * event.delta
        else:
            delta = -1 * (event.delta // 120)
        self.result_tree.yview_scroll(delta, "units")

    def resize_image(self, image):
        w, h = image.size
        scale_w = self.max_image_width / w if w > self.max_image_width else 1
        scale_h = self.max_image_height / h if h > self.max_image_height else 1
        scale = min(scale_w, scale_h)
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            return image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        return image

    def update_threshold(self, event=None):
        self.threshold1 = int(self.threshold1_var.get())
        self.threshold2 = int(self.threshold2_var.get())

def main():
    root = tk.Tk()
    app = BusinessCardExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
