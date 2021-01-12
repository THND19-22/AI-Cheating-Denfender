import datetime
import sys
import math
from typing import List, Tuple, Union

import cv2
from PyQt5.QtCore import pyqtSlot, QTimer, QDate
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QDialog, QApplication
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
                # print(str(r) + " " + str(g) + " " + str(idx_to_coordinates[start_idx][1]))
                cv2.line(image, idx_to_coordinates[start_idx][0],
                         idx_to_coordinates[end_idx][0], (0, g, r),
                         1)
    for landmark_px in idx_to_coordinates.values():
        # print(str(round(255 * landmark_px[1])) + " " + str(255 - round(255 * landmark_px[1])) + " " + str(landmark_px[1]))
        cv2.circle(image, landmark_px[0], 1, (0, round(255 * landmark_px[1]), 255 - round(255 * landmark_px[1])), 2)


class UiOutputDialog(QDialog):
    def __init__(self):
        """ Khởi động ứng dụng """
        super(UiOutputDialog, self).__init__()
        self.capture = None
        self.timer = QTimer(self)
        loadUi("./window.ui", self)
        self.holistic = mp.solutions.holistic.Holistic(upper_body_only=True)

        now = QDate.currentDate()
        current_date = now.toString('ddd dd MMMM yyyy')
        current_time = datetime.datetime.now().strftime("%I:%M %p")
        self.Date_Label.setText(current_date)
        self.Time_Label.setText(current_time)

        self.print_results = self.Print_Results.isChecked()
        self.Print_Results.stateChanged.connect(lambda: self.handle_print_results_change(self.Print_Results))

        self.image = None

    def handle_print_results_change(self, button):
        self.print_results = button.isChecked()

    @pyqtSlot()
    def start_video(self, camera_name):
        """
        Khởi động camera
        :param camera_name: id camera sẽ dùng
        """
        self.capture = camera_name

        self.timer.timeout.connect(self.update_frame)
        self.timer.start(10)

    def face_rec(self, image):
        """
        Nhận diện dáng người và tay từ ảnh đã cho
        :param image: ảnh từ camera
        :return: image: ảnh đuọc xử lý sau khi nhận diện
        """

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Bắt đầu nhận diện
        results = self.holistic.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.print_results:
            # Vẽ dáng người
            draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

            # Vẽ tay
            draw_landmarks(image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            draw_landmarks(image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            pass

        # Tính toán góc của khuôn mặt đối với camera
        if results.pose_landmarks is not None:
            chin_x = (results.pose_landmarks.landmark[9].x + results.pose_landmarks.landmark[10].x) / 2
            chin_y = (results.pose_landmarks.landmark[9].y + results.pose_landmarks.landmark[10].y) - \
                     results.pose_landmarks.landmark[0].y
            # print(results.pose_landmarks.landmark[0].visibility)
            nose_tip = normalize_to_pixel_coordinates(results.pose_landmarks.landmark[0].x,
                                                      results.pose_landmarks.landmark[0].y, image.shape[1],
                                                      image.shape[0])
            chin = normalize_to_pixel_coordinates(chin_x, chin_y, image.shape[1], image.shape[0])
            left_eye_outer = normalize_to_pixel_coordinates(results.pose_landmarks.landmark[3].x,
                                                            results.pose_landmarks.landmark[3].y, image.shape[1],
                                                            image.shape[0])
            right_eye_outer = normalize_to_pixel_coordinates(results.pose_landmarks.landmark[6].x,
                                                             results.pose_landmarks.landmark[6].y, image.shape[1],
                                                             image.shape[0])
            mouth_left = normalize_to_pixel_coordinates(results.pose_landmarks.landmark[9].x,
                                                        results.pose_landmarks.landmark[9].y, image.shape[1],
                                                        image.shape[0])
            mouth_right = normalize_to_pixel_coordinates(results.pose_landmarks.landmark[10].x,
                                                         results.pose_landmarks.landmark[10].y, image.shape[1],
                                                         image.shape[0])

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

            focal_length = image.shape[1]
            camera_matrix = np.array(
                [[focal_length, 0, image.shape[1] / 2],
                 [0, focal_length, image.shape[0] / 2],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

            rotation_mat, _ = cv2.Rodrigues(rotation_vector)
            pose_mat = cv2.hconcat((rotation_mat, translation_vector))
            _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
            # print(euler_angles)

            if self.print_results:
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeffs)
                '''
                for p in image_points:
                    cv2.circle(image, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)
                '''

                cv2.line(image, (int(image_points[0][0]), int(image_points[0][1])),
                         (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])), (255, 0, 0), 2)
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

    def set_warnings(self, ):
        """

        :return:
        """
        self.Warnings_List.addItem("1: Cảnh cáo - Đầu di chuyển hơn 60 độ so với camera!")
        self.Warnings_List.item(0).setText("abc")


# Bắt đầu ứng dụng
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UiOutputDialog()
    window.show()
    window.start_video(cv2.VideoCapture(0))
    window.set_warnings()
    sys.exit(app.exec())