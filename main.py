import datetime
import sys
import math
from typing import List, Tuple, Union

import cv2
from PyQt5.QtCore import QTimer, QDate, QDateTime
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtWidgets import QDialog, QApplication, QTreeWidgetItem
from PyQt5.uic import loadUi
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2

PRESENCE_THRESHOLD = 0.5
RGB_CHANNELS = 3
RED_COLOR = (0, 0, 255)
VISIBILITY_THRESHOLD = 0.5

# Các điểm thiết yếu của model 3D của khuôn mặt để tính toán góc khuôn mặt
model_points = np.array([
    (0.0, 0.0, 0.0),  # Mũi
    (0.0, -330.0, -65.0),  # Cằm
    (-225.0, 170.0, -135.0),  # Mắt trái
    (225.0, 170.0, -135.0),  # Mắt phải
    (-150.0, -150.0, -125.0),  # Miệng trái
    (150.0, -150.0, -125.0)  # Miệng phải
])


def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))


def normalize_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int,
                                   image_height: int) -> Union[None, Tuple[int, int]]:
    """ Code chuyển đổi dữ liệu đã cho thành tọa độ. """

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def draw_landmarks(image: np.ndarray,
                   landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: List[Tuple[int, int]] = None):
    """
    Vẽ dữ liệu đã nhận diện đuọc
    :param image: ảnh
    :param landmark_list: các điểm dữ liệu
    :param connections: các đoạn kết nối các điểm dữ liệu trên
    """
    if not landmark_list:
        return
    if image.shape[2] != RGB_CHANNELS:
        raise ValueError('Input image must contain three channel rgb data.')
    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and landmark.presence < PRESENCE_THRESHOLD)):
            continue
        vision = 1
        if landmark.HasField('visibility'):
            vision = landmark.visibility
        landmark_px = normalize_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = (landmark_px, vision)
    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Vẽ đoạn kết nối nếu điểm đầu và điểm cuối có thể thấy được.
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                continue
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                r = round((255 - round(255 * idx_to_coordinates[start_idx][1]) + 255 - round(
                    255 * idx_to_coordinates[end_idx][1])) / 2)
                g = round((round(255 * idx_to_coordinates[start_idx][1]) + round(
                    255 * idx_to_coordinates[end_idx][1])) / 2)
                cv2.line(image, idx_to_coordinates[start_idx][0],
                         idx_to_coordinates[end_idx][0], (0, g, r),
                         1)
    for landmark_px in idx_to_coordinates.values():
        cv2.circle(image, landmark_px[0], 1, (0, round(255 * landmark_px[1]), 255 - round(255 * landmark_px[1])), 1)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


class UISettings(QDialog):
    def __init__(self):
        super(UISettings, self).__init__()
        loadUi("settings.ui", self)

        self.print_results = self.Print_Results.isChecked()
        self.head_angle_limit = int(self.Head_Angle_Limit.text())

        self.Confirm_Button.accepted.connect(lambda: self.handle_accept_changes())
        self.Confirm_Button.rejected.connect(lambda: window.start_video())

    def handle_accept_changes(self):
        self.print_results = self.Print_Results.isChecked()
        self.head_angle_limit = int(self.Head_Angle_Limit.text())
        window.start_video()


class UIWarnings(QDialog):
    def __init__(self):
        super(UIWarnings, self).__init__()
        loadUi("warnings.ui", self)

        self.warning_list = []
        self.Save_Warnings.clicked.connect(lambda: self.handle_write_to_file())

        self.Confirm_Button.accepted.connect(lambda: window.start_video())
        self.Confirm_Button.rejected.connect(lambda: window.start_video())

    def handle_show(self, warnings):
        self.warning_list = warnings
        for warning in warnings:
            self.Warnings_List.addItem(warning)
        self.open()

    def handle_write_to_file(self):
        with open("log.txt", "wt", encoding='utf-8') as log:
            for warning in self.warning_list:
                log.write(warning + "\n")


class UiOutputDialog(QDialog):
    def __init__(self):
        """ Khởi động ứng dụng """
        super(UiOutputDialog, self).__init__()

        self.capture = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        loadUi("window.ui", self)

        self.holistic = mp.solutions.holistic.Holistic(upper_body_only=True)
        self.face = mp.solutions.face_mesh.FaceMesh(max_num_faces=100)
        self.hand = mp.solutions.hands.Hands(max_num_hands=100)

        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)

        self.image = None

        self.settings = UISettings()
        self.Settings_Button.clicked.connect(lambda: self.handle_setting_button())

        self.warnings = []
        self.prev_waring_code = -1
        self.prev_waring_code_1 = -1
        self.prev_waring_code_2 = -1
        self.warning_history = UIWarnings()
        self.Warning_History.clicked.connect(lambda: self.handle_waring_history_button())

        self.net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
        with open("yolov3.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def handle_waring_history_button(self):
        self.pause_video()
        self.warning_history.handle_show(self.warnings)

    def handle_setting_button(self):
        self.pause_video()
        self.settings.show()

    def init_video(self, camera):
        self.capture = camera

    def start_video(self):
        self.timer.start(1)

    def pause_video(self):
        self.timer.stop()

    def face_rec(self, image):
        """
        Nhận diện dáng người và tay từ ảnh đã cho
        :param image: ảnh từ camera
        :return: image: ảnh đuọc xử lý sau khi nhận diện
        """
        width = image.shape[1]
        height = image.shape[0]

        raw_img = image

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Bắt đầu nhận diện
        faces = self.face.process(image)
        hands = self.hand.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if not faces.multi_face_landmarks:
            return image

        count = 0
        for face_landmarks in faces.multi_face_landmarks:
            if self.settings.print_results:
                draw_landmarks(image, face_landmarks, mp.solutions.face_mesh.FACE_CONNECTIONS)
            """
            nose_tip 1
            chin = 199
            left_eye_outer = 249
            right_eye_outer = 7
            mouth_left = 291
            mouth_right = 61
            """
            nose_tip = normalize_to_pixel_coordinates(face_landmarks.landmark[1].x,
                                                      face_landmarks.landmark[1].y, width, height)
            chin = normalize_to_pixel_coordinates(face_landmarks.landmark[199].x,
                                                  face_landmarks.landmark[199].y, width, height)
            left_eye_outer = normalize_to_pixel_coordinates(face_landmarks.landmark[249].x,
                                                            face_landmarks.landmark[249].y, width, height)
            right_eye_outer = normalize_to_pixel_coordinates(face_landmarks.landmark[7].x,
                                                             face_landmarks.landmark[7].y, width, height)
            mouth_left = normalize_to_pixel_coordinates(face_landmarks.landmark[291].x,
                                                        face_landmarks.landmark[291].y, width, height)
            mouth_right = normalize_to_pixel_coordinates(face_landmarks.landmark[61].x,
                                                         face_landmarks.landmark[61].y, width, height)

            if chin is None or left_eye_outer is None or right_eye_outer is None or mouth_left is None or mouth_right is None:
                return image

            image_points = np.array([
                nose_tip,
                chin,
                left_eye_outer,
                right_eye_outer,
                mouth_left,
                mouth_right
            ], dtype="double")

            focal_length = width
            camera_matrix = np.array(
                [[focal_length, 0, width / 2],
                 [0, focal_length, height / 2],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            if self.Warnings_List.topLevelItemCount() <= count:
                item = QTreeWidgetItem(["Thí sinh #" + str(count + 1)])
                item.setExpanded(True)
                item.addChildren([QTreeWidgetItem(), QTreeWidgetItem()])
                self.Warnings_List.addTopLevelItem(item)

            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            angle = round(abs(euler_angles[1][0]), 2)
            self.Warnings_List.topLevelItem(count).child(0).setText(0, "Đầu di chuyển ngang " + str(
                angle) + " độ so với camera")
            color = int(angle / self.settings.head_angle_limit * 255)
            if color > 255:
                if self.prev_waring_code_1 != 3:
                    self.warnings.append(QDateTime.currentDateTime().toString() + ": Đầu di chuyển ngang " + str(
                        angle) + " độ so với camera")
                    self.prev_waring_code_1 = 3
                color = 255
            self.Warnings_List.topLevelItem(count).child(0).setBackground(0, QColor(color, 255 - color, 0))

            angle = round(abs(euler_angles[2][0]), 2)
            self.Warnings_List.topLevelItem(count).child(1).setText(0, "Đầu di chuyển dọc " + str(
                angle) + " độ so với camera")
            color = int(angle / self.settings.head_angle_limit * 255)
            if color > 255:
                if self.prev_waring_code_2 != 5:
                    self.warnings.append(
                        QDateTime.currentDateTime().toString() + ": Đầu di chuyển dọc " + str(
                            angle) + " độ so với camera")
                    self.prev_waring_code_2 = 5
                color = 255
            self.Warnings_List.topLevelItem(count).child(1).setBackground(0, QColor(color, 255 - color, 0))

            if self.settings.print_results:
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeffs)
                '''
                for p in image_points:
                    cv2.circle(image, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
                '''

                cv2.line(image, (int(image_points[0][0]), int(image_points[0][1])),
                         (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])), (255, 0, 0), 2)
            count += 1

        while self.Warnings_List.topLevelItemCount() > len(faces.multi_face_landmarks):
            self.Warnings_List.takeTopLevelItem(len(faces.multi_face_landmarks))

        if not hands.multi_hand_landmarks:
            for i in range(len(faces.multi_face_landmarks)):
                if self.Warnings_List.topLevelItem(i).childCount() == 2:
                    self.Warnings_List.topLevelItem(i).addChild(QTreeWidgetItem())
                self.Warnings_List.topLevelItem(i).child(2).setText(0, "Không tìm thấy tay!")
                self.Warnings_List.topLevelItem(i).child(2).setBackground(0, QColor("red"))
                if self.prev_waring_code != 0:
                    self.warnings.append(QDateTime.currentDateTime().toString() + ": Không tìm thấy tay!")
                    self.prev_waring_code = 0
            return image

        face_list = {}
        for hand_landmarks in hands.multi_hand_landmarks:
            if self.settings.print_results:
                draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

            hand_x = hand_landmarks.landmark[0].x
            hand_y = hand_landmarks.landmark[0].y

            best_face = -1
            best_face_value = 1

            count = 0
            for face_landmarks in faces.multi_face_landmarks:
                if count in face_list and face_list[count] == 2:
                    count += 1
                    continue
                face_x = face_landmarks.landmark[0].x
                face_y = face_landmarks.landmark[0].y
                dist = math.sqrt(pow(abs(face_x - hand_x), 2) + pow(abs(face_y - hand_y), 2))
                if dist < best_face_value:
                    best_face_value = dist
                    best_face = count
                count += 1
            if best_face != -1:
                if best_face in face_list:
                    face_list[best_face] = 2
                else:
                    face_list[best_face] = 1

        for i in range(len(faces.multi_face_landmarks)):
            if self.Warnings_List.topLevelItem(i).childCount() == 2:
                self.Warnings_List.topLevelItem(i).addChild(QTreeWidgetItem())
            if i in face_list:
                if face_list[i] == 1:
                    self.Warnings_List.topLevelItem(i).child(2).setText(0, "Chỉ phát hiện được một tay!")
                    self.Warnings_List.topLevelItem(i).child(2).setBackground(0, QColor("yellow"))
                    if self.prev_waring_code != 1:
                        self.warnings.append(QDateTime.currentDateTime().toString() + ": Chỉ phát hiện được một tay!")
                        self.prev_waring_code = 1
                else:
                    if self.Warnings_List.topLevelItem(i).childCount() == 3:
                        self.Warnings_List.topLevelItem(i).takeChild(2)
            else:
                self.Warnings_List.topLevelItem(i).child(2).setText(0, "Không tìm thấy tay!")
                self.Warnings_List.topLevelItem(i).child(2).setBackground(0, QColor("red"))
                if self.prev_waring_code != 0:
                    self.warnings.append(QDateTime.currentDateTime().toString() + ": Không tìm thấy tay!")
                    self.prev_waring_code = 0
        """
        blob = cv2.dnn.blobFromImage(raw_img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(get_output_layers(self.net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            color = self.colors[class_ids[i]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, str(self.classes[class_ids[i]]), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,
                        2)
        """
        return image

    def update_frame(self):
        """
        Cập nhật hình ảnh lên ứng dụng
        """
        ret, self.image = self.capture.read()
        self.display_image(self.image, 1)

    def display_image(self, image, windowed=1):
        """
        Phát hình ảnh đã được xử lý lên ứng dụng
        :param image: ảnh từ camera
        :param windowed: số window đang hiện
        """
        image = cv2.resize(image, (640, 480))
        image = cv2.flip(image, 1)
        try:
            image = self.face_rec(image)
        except Exception as e:
            print(e)
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        out_image = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        out_image = out_image.rgbSwapped()

        if windowed == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(out_image))
            self.imgLabel.setScaledContents(True)


# Bắt đầu ứng dụng
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UiOutputDialog()
    window.show()
    window.init_video(cv2.VideoCapture(0))
    window.start_video()
    sys.exit(app.exec())
