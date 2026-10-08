import cv2
import mediapipe as mp


class HandDetector:

    def __init__(
        self,
        max_num_hands=2,
        detection_confidence=0.5,
        tracking_confidence=0.5,
        model_complexity=0
    ):

        self.mp_hands = (
            mp.solutions.hands
        )

        self.mp_drawing = (
            mp.solutions.drawing_utils
        )


        self.hands = (
            self.mp_hands.Hands(

                static_image_mode=False,

                max_num_hands=(
                    max_num_hands
                ),

                model_complexity=(
                    model_complexity
                ),

                min_detection_confidence=(
                    detection_confidence
                ),

                min_tracking_confidence=(
                    tracking_confidence
                )
            )
        )


    # =====================================================
    # DETECT
    # =====================================================

    def detect(
        self,
        frame
    ):

        """
        Nhận frame BGR từ OpenCV
        và trả về kết quả MediaPipe.
        """

        if frame is None:

            return None


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        rgb_frame.flags.writeable = (
            False
        )


        results = self.hands.process(
            rgb_frame
        )


        rgb_frame.flags.writeable = (
            True
        )


        return results


    # =====================================================
    # GET LEFT / RIGHT
    # =====================================================

    def get_hands(
        self,
        results
    ):

        """
        Return:

        {
            "Left": landmark hoặc None,
            "Right": landmark hoặc None
        }
        """

        hands_data = {

            "Left": None,

            "Right": None
        }


        if results is None:

            return hands_data


        if (
            not results.multi_hand_landmarks
            or
            not results.multi_handedness
        ):

            return hands_data


        for (
            hand_landmarks,
            handedness
        ) in zip(

            results.multi_hand_landmarks,

            results.multi_handedness

        ):

            classification = (
                handedness
                .classification[0]
            )


            label = (
                classification.label
            )


            if label in hands_data:

                hands_data[label] = (
                    hand_landmarks
                )


        return hands_data


    # =====================================================
    # HAND COUNT
    # =====================================================

    def count_hands(
        self,
        results
    ):

        if (
            results is None
            or
            not results.multi_hand_landmarks
        ):

            return 0


        return len(
            results.multi_hand_landmarks
        )


    # =====================================================
    # HAS HAND
    # =====================================================

    def has_hand(
        self,
        results
    ):

        return (
            self.count_hands(
                results
            )
            > 0
        )


    # =====================================================
    # DRAW
    # =====================================================

    def draw_hands(
        self,
        frame,
        results
    ):

        """
        Vẽ landmark cho tất cả bàn tay.
        """

        if frame is None:

            return frame


        if (
            results is None
            or
            not results.multi_hand_landmarks
        ):

            return frame


        for hand_landmarks in (
            results.multi_hand_landmarks
        ):

            self.mp_drawing.draw_landmarks(

                frame,

                hand_landmarks,

                self.mp_hands.HAND_CONNECTIONS
            )


        return frame


    # =====================================================
    # CLOSE
    # =====================================================

    def close(self):

        if self.hands is not None:

            self.hands.close()