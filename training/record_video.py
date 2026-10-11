import os
import cv2
import time


# =========================================
# CONFIG
# =========================================

VIDEO_DATASET_DIR = "video_dataset"

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

FPS = 30

RECORD_SECONDS = 3


# =========================================
# INPUT LABEL
# =========================================

label = input(
    "Nhap label (VD: XIN_CHAO): "
).strip().upper()


if not label:

    print("Label khong hop le.")
    exit()


save_dir = os.path.join(
    VIDEO_DATASET_DIR,
    label
)


os.makedirs(
    save_dir,
    exist_ok=True
)


# =========================================
# FIND NEXT VIDEO INDEX
# =========================================

existing_files = [

    file

    for file in os.listdir(
        save_dir
    )

    if file.endswith(".mp4")
]


video_index = (
    len(existing_files) + 1
)


# =========================================
# CAMERA
# =========================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    VIDEO_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    VIDEO_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    FPS
)


if not cap.isOpened():

    print(
        "Khong the mo camera."
    )

    exit()


print()
print(
    "=============================="
)
print(
    "THU VIDEO KY HIEU DONG"
)
print(
    "=============================="
)
print(
    "Label:",
    label
)
print()
print(
    "SPACE: quay video"
)
print(
    "Q: thoat"
)
print()


# =========================================
# MAIN LOOP
# =========================================

while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame = cv2.flip(
        frame,
        1
    )


    # =====================================
    # UI
    # =====================================

    cv2.putText(
        frame,
        f"Label: {label}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        (
            f"Video: "
            f"{video_index}"
        ),
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "SPACE: Record",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q: Quit",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.imshow(
        "Record Dynamic Sign Video",
        frame
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # =====================================
    # QUIT
    # =====================================

    if key == ord("q"):

        break


    # =====================================
    # RECORD VIDEO
    # =====================================

    elif key == ord(" "):

        filename = (
            f"video_"
            f"{video_index:04d}"
            f".mp4"
        )


        filepath = os.path.join(
            save_dir,
            filename
        )


        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )


        writer = cv2.VideoWriter(

            filepath,

            fourcc,

            FPS,

            (
                VIDEO_WIDTH,
                VIDEO_HEIGHT
            )
        )


        print()
        print(
            "Chuan bi quay..."
        )


        # =================================
        # COUNTDOWN
        # =================================

        for count in [
            3,
            2,
            1
        ]:

            start_time = time.time()


            while (
                time.time()
                - start_time
                < 1
            ):

                ret, countdown_frame = (
                    cap.read()
                )


                if not ret:

                    continue


                countdown_frame = (
                    cv2.flip(
                        countdown_frame,
                        1
                    )
                )


                cv2.putText(
                    countdown_frame,
                    str(count),
                    (
                        VIDEO_WIDTH // 2 - 30,
                        VIDEO_HEIGHT // 2
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3,
                    (0, 0, 255),
                    5
                )


                cv2.imshow(
                    "Record Dynamic Sign Video",
                    countdown_frame
                )


                cv2.waitKey(1)


        print(
            "Dang quay:",
            filepath
        )


        # =================================
        # RECORD
        # =================================

        record_start = (
            time.time()
        )


        while (
            time.time()
            - record_start
            < RECORD_SECONDS
        ):

            ret, record_frame = (
                cap.read()
            )


            if not ret:

                continue


            record_frame = cv2.flip(
                record_frame,
                1
            )


            writer.write(
                record_frame
            )


            elapsed = (
                time.time()
                - record_start
            )


            cv2.putText(
                record_frame,
                "RECORDING",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


            cv2.putText(
                record_frame,
                (
                    f"{elapsed:.1f}"
                    f"/{RECORD_SECONDS}s"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.imshow(
                "Record Dynamic Sign Video",
                record_frame
            )


            cv2.waitKey(1)


        writer.release()


        print(
            "Da luu:",
            filepath
        )


        video_index += 1


# =========================================
# CLEANUP
# =========================================

cap.release()

cv2.destroyAllWindows()

print(
    "Da dong chuong trinh."
)