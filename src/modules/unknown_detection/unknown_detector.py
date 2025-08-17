import numpy as np
import cv2

class UnknownDetector:
    def __init__(self):
        pass


    def _normalize_to_0_255(self, img):
        normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    def _find_contours(self, depth_map):
        _, thresh = cv2.threshold(depth_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = np.zeros_like(depth_map)
        cv2.drawContours(contour_img, contours, -1, (255), 2)
        return contour_img

    def _detect_objects_from_binary_contour(self, contour_img, edge_margin=0.2, box_h_threshold=10):
        img_height, img_width = contour_img.shape[:2]
        _, contour_img = cv2.threshold(contour_img, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(contour_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, False

        main_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(main_contour)

        main_contour_w_percentage = w / img_width * 100
        mask2 = np.zeros_like(contour_img)

        if main_contour_w_percentage > 75:
            section_width = w // 4
            mask2[:, x + section_width: x + (3 * section_width)] = 255
        elif main_contour_w_percentage > 60:
            section_width = w // 3
            mask2[:, x + section_width: x + (2 * section_width)] = 255
        else:
            mask2[:, x: x + w] = 255

        x_left_bound = int(img_width * edge_margin)
        x_right_bound = int(img_width * (1 - edge_margin))

        res2 = cv2.bitwise_and(contour_img, contour_img, mask=mask2)
        contours2, _ = cv2.findContours(res2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        exist_unknown_object = False
        bbox = None
        if contours2:
            largest_contour2 = max(contours2, key=cv2.contourArea)
            bbox = cv2.boundingRect(largest_contour2)
            x2, y2, w2, h2 = bbox
            h2_percentage = h2 / img_height * 100

            x_center = (x2 + w2 / 2)

            if (x_left_bound <= x_center <= x_right_bound) and h2_percentage >= box_h_threshold:
                exist_unknown_object = True

        return exist_unknown_object, bbox

    def has_unknown_object(self, depth_map, edge_margin=0.2, box_h_threshold=10):
        # Assuming depth_map is already normalized to 0-255 as expected by _find_contours
        normalized_img = self._normalize_to_0_255(depth_map)
        contour_img = self._find_contours(normalized_img)
        exist_unknown_object, bbox =  self._detect_objects_from_binary_contour(contour_img, edge_margin, box_h_threshold)

        return exist_unknown_object, bbox, contour_img

    def has_unknown_objects_in_depth_frames(self, depth_frames, edge_margin=0.2, box_h_threshold=10):

        for frame_id, depth_map in depth_frames.items():
            # Assuming depth_map is already normalized to 0-255 as expected by _find_contours
            normalized_img = self._normalize_to_0_255(depth_map)
            contour_img = self._find_contours(normalized_img)
            exists, bbox = self._detect_objects_from_binary_contour(contour_img, edge_margin, box_h_threshold)
            if exists:
                return True, bbox, contour_img, frame_id
        return False, None, None, None


