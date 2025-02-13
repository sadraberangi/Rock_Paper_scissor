import cv2
import numpy as np
import time
import mediapipe as mp
from ultralytics import YOLO

mp_face_detection = mp.solutions.face_detection

mp_face_detection = mp.solutions.face_detection

def show_winner_animation(
    cap,
    is_p1_winner=False,
    is_p2_winner=False,
    crown_image_path='crown.png',
    display_seconds=3,
    margin_ratio=0.9
):
    """
    1) Reads frames from 'cap' for 'display_seconds'.
    2) Detects winner's face bounding box (P1 => left side, P2 => right side).
    3) Expands that box by a margin around the face (margin_ratio).
    4) Crops the region, then scales it to the full window/frame size (like a zoom).
    5) Places a crown ABOVE the scaled face in the output image.
    6) Displays the result in a window titled "Winner Animation".

    :param cap: cv2.VideoCapture (already opened).
    :param is_p1_winner: If True, find face on the LEFT side of the frame.
    :param is_p2_winner: If True, find face on the RIGHT side of the frame.
    :param crown_image_path: Path to a crown PNG (ideally with alpha channel).
    :param display_seconds: How long to show the animation (seconds).
    :param margin_ratio: How much extra space around the face bounding box (e.g. 0.3 = 30%).
    """

    # Load the crown image (4-channel if possible)
    crown = cv2.imread(crown_image_path, cv2.IMREAD_UNCHANGED)
    if crown is None:
        print(f"Error loading crown image: {crown_image_path}")
        return

    # Initialize Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        start_time = time.time()
        
        while (time.time() - start_time) < display_seconds:
            ret, frame = cap.read()
            if not ret:
                break

            # For final display size
            out_h, out_w = frame.shape[:2]  # The original frame dimensions

            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results and results.detections:
                # Pick the largest face bounding box on the correct side
                biggest_area = 0
                best_bbox = None

                frame_center_x = out_w // 2

                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    x_min = int(bboxC.xmin * out_w)
                    y_min = int(bboxC.ymin * out_h)
                    w_box = int(bboxC.width * out_w)
                    h_box = int(bboxC.height * out_h)
                    x_max = x_min + w_box
                    y_max = y_min + h_box

                    area = w_box * h_box
                    face_center_x = (x_min + x_max) // 2

                    # Left side => Player 1
                    if is_p1_winner and face_center_x <= frame_center_x:
                        if area > biggest_area:
                            biggest_area = area
                            best_bbox = (x_min, y_min, x_max, y_max)

                    # Right side => Player 2
                    elif is_p2_winner and face_center_x > frame_center_x:
                        if area > biggest_area:
                            biggest_area = area
                            best_bbox = (x_min, y_min, x_max, y_max)

                if best_bbox:
                    x_min, y_min, x_max, y_max = best_bbox
                    face_w = x_max - x_min
                    face_h = y_max - y_min

                    # Expand bounding box by `margin_ratio`
                    margin_w = int(face_w * margin_ratio)
                    margin_h = int(face_h * margin_ratio)

                    crop_x_min = max(0, x_min - margin_w)
                    crop_y_min = max(0, y_min - margin_h)
                    crop_x_max = min(out_w, x_max + margin_w)
                    crop_y_max = min(out_h, y_max + margin_h)

                    cropped = frame[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

                    # Face coords inside 'cropped' (we'll need for crown placement)
                    face_x_min = x_min - crop_x_min
                    face_y_min = y_min - crop_y_min
                    face_x_max = x_max - crop_x_min
                    face_y_max = y_max - crop_y_min
                    face_w_cropped = face_x_max - face_x_min
                    face_h_cropped = face_y_max - face_y_min

                    # -- Scale the cropped region to fill the entire original frame size --
                    final = cv2.resize(cropped, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                    # Compute scale factors
                    crop_w = (crop_x_max - crop_x_min)
                    crop_h = (crop_y_max - crop_y_min)
                    scale_x = out_w / float(crop_w)
                    scale_y = out_h / float(crop_h)

                    # Face coords in scaled image
                    scaled_face_x_min = int(face_x_min * scale_x)
                    scaled_face_y_min = int(face_y_min * scale_y)
                    scaled_face_x_max = int(face_x_max * scale_x)
                    scaled_face_y_max = int(face_y_max * scale_y)
                    scaled_face_w = scaled_face_x_max - scaled_face_x_min

                    # Place the crown above the scaled face
                    crown_aspect = crown.shape[1] / crown.shape[0]  # w/h
                    final_crown_w = scaled_face_w
                    final_crown_h = int(final_crown_w / crown_aspect)

                    # Resize the crown
                    crown_resized = cv2.resize(crown, (final_crown_w, final_crown_h))

                    # Offset above the face
                    offset_px = 200
                    crown_x_min = scaled_face_x_min
                    crown_y_min = scaled_face_y_min - final_crown_h - offset_px

                    # Center horizontally over the face
                    shift_x = (scaled_face_w - final_crown_w) // 2
                    crown_x_min += shift_x

                    # Clamp in final image
                    if crown_y_min < 0:
                        crown_y_min = 0

                    # Overlay the crown on 'final'
                    overlay_image_with_alpha(final, crown_resized, crown_x_min, crown_y_min)

                    # Optionally write "WINNER!"
                    cv2.putText(
                        final,
                        "WINNER!",
                        (scaled_face_x_min, scaled_face_y_min - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )

                    cv2.imshow("Winner Animation", final)
                else:
                    # If we didn't find the face for the winner, just show original frame
                    cv2.imshow("Winner Animation", frame)
            else:
                # No detections => show original frame
                cv2.imshow("Winner Animation", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

def overlay_image_with_alpha(background, overlay, x, y):
    """
    Overlays 'overlay' (potentially with alpha channel) onto 'background'
    at position (x,y). 
    """
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    y1, y2 = y, y + oh
    x1, x2 = x, x + ow

    # Quick out-of-bounds check
    if x1 >= bw or y1 >= bh or x2 < 0 or y2 < 0:
        return

    # Clip if partially outside
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(bw, x2)
    y2_clip = min(bh, y2)

    overlay_x = x1_clip - x1
    overlay_y = y1_clip - y1
    overlay_w = x2_clip - x1_clip
    overlay_h = y2_clip - y1_clip

    # If overlay has alpha channel
    if overlay.shape[2] == 4:
        overlay_roi = overlay[overlay_y:overlay_y+overlay_h, overlay_x:overlay_x+overlay_w]
        overlay_rgb = overlay_roi[..., :3]
        alpha = overlay_roi[..., 3] / 255.0

        bg_roi = background[y1_clip:y1_clip+overlay_h, x1_clip:x1_clip+overlay_w]

        for c in range(3):
            bg_roi[..., c] = (overlay_rgb[..., c] * alpha +
                              bg_roi[..., c] * (1.0 - alpha))
    else:
        # No alpha => opaque
        overlay_roi = overlay[overlay_y:overlay_y+overlay_h, overlay_x:overlay_x+overlay_w]
        background[y1_clip:y1_clip+overlay_h, x1_clip:x1_clip+overlay_w] = overlay_roi


def apply_red_mask(frame, x_min, y_min, x_max, y_max, alpha=0.4):
    """
    Overlays a semi-transparent red rectangle on the face region.
    (x_min, y_min), (x_max, y_max) are the bounding box corners.
    """
    h, w, _ = frame.shape
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 255), -1)  # solid red
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def apply_mask_if_cheater(frame, p1_mask, p2_mask):
    """
    Runs MediaPipe face detection on the current 'frame'.
    If p1_mask == True, we overlay a red mask on the LEFT-side face (Player 1).
    If p2_mask == True, we overlay a red mask on the RIGHT-side face (Player 2).
    """
    if not (p1_mask or p2_mask):
        return  # No one is masked => no need for face detection

    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Convert the BGR image to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            frame_center_x = frame.shape[1] // 2
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                x_min = int(bboxC.xmin * frame.shape[1])
                y_min = int(bboxC.ymin * frame.shape[0])
                w_box = int(bboxC.width * frame.shape[1])
                h_box = int(bboxC.height * frame.shape[0])

                x_max = x_min + w_box
                y_max = y_min + h_box

                face_center_x = (x_min + x_max) // 2

                # Left side => Player 1
                if face_center_x <= frame_center_x and p1_mask:
                    apply_red_mask(frame, x_min, y_min, x_max, y_max, alpha=0.4)
                    cv2.putText(
                        frame,
                        "CHEATER",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                # Right side => Player 2
                elif face_center_x > frame_center_x and p2_mask:
                    apply_red_mask(frame, x_min, y_min, x_max, y_max, alpha=0.4)
                    cv2.putText(
                        frame,
                        "CHEATER",
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )


# ------------------- YOLO Helpers ------------------- #
def draw_bounding_box(frame, box, label, color=(0, 255, 0)):
    if not box:
        return
    x_min, y_min, x_max, y_max = box
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(
        frame,
        label,
        (x_min, y_min - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

def detect_players(frame, result, class_names):
    """
    Now Player 1 is on the LEFT side (center_x <= frame_center_x),
    Player 2 is on the RIGHT side (center_x > frame_center_x).
    """
    p1_box, p2_box = None, None
    p1_label, p2_label = "Unknown", "Unknown"
    p1_conf, p2_conf = -1, -1
    p1_center, p2_center = None, None

    frame_center_x = frame.shape[1] // 2
    for box in result.boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        confidence = float(box.conf[0])
        cls_idx = int(box.cls[0])
        label = class_names.get(cls_idx, "Unknown")

        # LEFT side => Player 1
        if center_x <= frame_center_x:
            if confidence > p1_conf:
                p1_box = (x_min, y_min, x_max, y_max)
                p1_label = label
                p1_conf = confidence
                p1_center = (center_x, center_y)
        else:
            # RIGHT side => Player 2
            if confidence > p2_conf:
                p2_box = (x_min, y_min, x_max, y_max)
                p2_label = label
                p2_conf = confidence
                p2_center = (center_x, center_y)

    return p1_box, p2_box, p1_label, p2_label, p1_center, p2_center, p1_conf, p2_conf


# ------------------- Game Phases ------------------- #
def pre_play(cap, model, p1_mask, p2_mask, number_of_wins, player1_score, player2_score, wait_rock=7):
    """
    Wait until both players show 'Rock' simultaneously for 'wait_rock' consecutive frames.
    (Player 1 = left side, Player 2 = right side.)
    """
    class_names = model.names
    rock_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result_0 = results[0]

        p1_box, p2_box, p1_label, p2_label, *_ = detect_players(frame, result_0, class_names)

        # Draw bounding boxes
        draw_bounding_box(frame, p1_box, f'Player 1: {p1_label}', (0, 255, 0))
        draw_bounding_box(frame, p2_box, f'Player 2: {p2_label}', (255, 0, 0))

        if p1_label == "Rock" and p2_label == "Rock":
            rock_counter += 1
        else:
            rock_counter = 0

        cv2.putText(
            frame,
            f'Player 1 Score: {player1_score}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f'Player 2 Score: {player2_score}',
            (frame.shape[1] - 230, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            'GO ROCK!',
            (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2
        )

        # --- Apply mask if cheater ---
        apply_mask_if_cheater(frame, p1_mask, p2_mask)

        cv2.imshow('YOLO - Players Detection', frame)

        if cv2.waitKey(1) == 27 or rock_counter == wait_rock:
            break


def counting(cap, model, p1_mask, p2_mask, number_of_wins, player1_score, player2_score):
    """
    3-second countdown, ensuring both players remain on 'Rock' (or unknown)
    and do not move suspiciously. If suspicious => they get flagged as cheater.
    """
    class_names = model.names
    countdown_seconds = 3
    start_time = time.time()

    movement_threshold = 5
    max_suspicious_streak = 5

    prev_player1_pos = None
    prev_player2_pos = None

    p1_suspicious_streak = 0
    p2_suspicious_streak = 0
    p1_cheated = False
    p2_cheated = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.time() - start_time
        remaining = countdown_seconds - elapsed

        if remaining <= 0:
            cv2.putText(
                frame,
                "GO!",
                (frame.shape[1] // 2 - 50, frame.shape[0] // 2 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                2
            )
            apply_mask_if_cheater(frame, p1_mask, p2_mask)
            cv2.imshow('YOLO - Players Detection', frame)
            cv2.waitKey(500)
            break

        results = model(frame)
        result_0 = results[0]

        p1_box, p2_box, p1_label, p2_label, p1_center, p2_center, *_ = detect_players(frame, result_0, class_names)

        draw_bounding_box(frame, p1_box, f'Player 1: {p1_label}', (0, 255, 0))
        draw_bounding_box(frame, p2_box, f'Player 2: {p2_label}', (255, 0, 0))

        # Suspicion logic: If you're not showing "Rock"/"Unknown" or you're not moving enough, mark suspicious.
        # Player 1
        is_p1_suspicious = False
        if p1_label not in ("Rock", "Unknown"):
            is_p1_suspicious = True
        else:
            if prev_player1_pos and p1_center:
                dx = abs(p1_center[0] - prev_player1_pos[0])
                dy = abs(p1_center[1] - prev_player1_pos[1])
                if dx < movement_threshold and dy < movement_threshold:
                    is_p1_suspicious = True

        if is_p1_suspicious:
            p1_suspicious_streak += 1
        else:
            p1_suspicious_streak = 0

        if p1_suspicious_streak >= max_suspicious_streak:
            p1_cheated = True

        # Player 2
        is_p2_suspicious = False
        if p2_label not in ("Rock", "Unknown"):
            is_p2_suspicious = True
        else:
            if prev_player2_pos and p2_center:
                dx = abs(p2_center[0] - prev_player2_pos[0])
                dy = abs(p2_center[1] - prev_player2_pos[1])
                if dx < movement_threshold and dy < movement_threshold:
                    is_p2_suspicious = True

        if is_p2_suspicious:
            p2_suspicious_streak += 1
        else:
            p2_suspicious_streak = 0

        if p2_suspicious_streak >= max_suspicious_streak:
            p2_cheated = True

        if p1_center:
            prev_player1_pos = p1_center
        if p2_center:
            prev_player2_pos = p2_center

        cv2.putText(
            frame,
            str(int(remaining)),
            (frame.shape[1] // 2 - 50, frame.shape[0] // 2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            2
        )

        if p1_cheated:
            cv2.putText(
                frame,
                'Player 1 Cheater!',
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        if p2_cheated:
            cv2.putText(
                frame,
                'Player 2 Cheater!',
                (frame.shape[1] // 2 + 20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # --- Apply mask if cheater ---
        apply_mask_if_cheater(frame, p1_mask, p2_mask)

        cv2.imshow('YOLO - Players Detection', frame)
        if cv2.waitKey(1) == 27:
            break

    return p1_cheated, p2_cheated


def finalize_and_check_decisions(cap, model, p1_mask, p2_mask,
                                 final_decision_lock_time=2.0,
                                 post_lock_check_time=2.0):
    """
    Locks each player's final label in the first phase (final_decision_lock_time).
    Then checks if it changes in the second phase (post_lock_check_time).
    If it changes => that player is flagged as cheater.
    """
    class_names = model.names

    # PHASE 1: Lock
    p1_best_conf = -1
    p2_best_conf = -1
    p1_final_label = "Unknown"
    p2_final_label = "Unknown"

    start_lock_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_lock = time.time() - start_lock_time
        remaining_lock = final_decision_lock_time - elapsed_lock

        if remaining_lock <= 0:
            break

        results = model(frame)
        det = results[0]

        p1_label_current = "Unknown"
        p2_label_current = "Unknown"
        p1_conf_current = -1
        p2_conf_current = -1

        frame_center = frame.shape[1] // 2
        for box in det.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = class_names.get(cls_idx, "Unknown")

            center_x = (x_min + x_max) // 2
            if center_x <= frame_center:  # LEFT => Player 1
                if conf > p1_conf_current:
                    p1_label_current = label
                    p1_conf_current = conf
            else:  # RIGHT => Player 2
                if conf > p2_conf_current:
                    p2_label_current = label
                    p2_conf_current = conf

        if p1_conf_current > p1_best_conf and p1_label_current != "Unknown":
            p1_best_conf = p1_conf_current
            p1_final_label = p1_label_current
        if p2_conf_current > p2_best_conf and p2_label_current != "Unknown":
            p2_best_conf = p2_conf_current
            p2_final_label = p2_label_current

        lock_msg = f"Locking final decisions... {int(remaining_lock)}s left"
        cv2.putText(
            frame,
            lock_msg,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f'P1 Final: {p1_final_label}',
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f'P2 Final: {p2_final_label}',
            (frame.shape[1] - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # --- Apply mask if cheater ---
        apply_mask_if_cheater(frame, p1_mask, p2_mask)

        cv2.imshow('Final Decision', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # PHASE 2: Cheating check
    p1_cheated = False
    p2_cheated = False

    start_check_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed_check = time.time() - start_check_time
        remaining_check = post_lock_check_time - elapsed_check
        if remaining_check <= 0:
            break

        results = model(frame)
        det = results[0]

        p1_label_current = "Unknown"
        p2_label_current = "Unknown"
        p1_conf_current = -1
        p2_conf_current = -1

        frame_center = frame.shape[1] // 2
        for box in det.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_idx = int(box.cls[0])
            label = class_names.get(cls_idx, "Unknown")

            center_x = (x_min + x_max) // 2
            if center_x <= frame_center:  # LEFT => Player 1
                if conf > p1_conf_current:
                    p1_label_current = label
                    p1_conf_current = conf
            else:  # RIGHT => Player 2
                if conf > p2_conf_current:
                    p2_label_current = label
                    p2_conf_current = conf

        # If a player's label changes => cheater
        if not p1_cheated and p1_label_current not in ("Unknown", p1_final_label):
            p1_cheated = True
        if not p2_cheated and p2_label_current not in ("Unknown", p2_final_label):
            p2_cheated = True

        if p1_cheated:
            cv2.putText(
                frame,
                'P1 CHEATED!',
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )
        if p2_cheated:
            cv2.putText(
                frame,
                'P2 CHEATED!',
                (frame.shape[1] - 250, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        check_msg = f"Verifying no changes... {int(remaining_check)}s left"
        cv2.putText(
            frame,
            check_msg,
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f'P1 Final: {p1_final_label}',
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            f'P2 Final: {p2_final_label}',
            (frame.shape[1] - 250, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # --- Apply mask if cheater ---
        apply_mask_if_cheater(frame, p1_mask, p2_mask)

        cv2.imshow('Final Decision', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    return p1_final_label, p2_final_label, p1_cheated, p2_cheated


def show_cheater_message(cap, p1_cheated, p2_cheated):
    ret, freeze_frame = cap.read()
    if not ret:
        return
    start_wait = time.time()
    while (time.time() - start_wait) < 2:
        frame_copy = freeze_frame.copy()
        if p1_cheated:
            cv2.putText(
                frame_copy,
                "Player 1 Cheated!",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        if p2_cheated:
            cv2.putText(
                frame_copy,
                "Player 2 Cheated!",
                (frame_copy.shape[1] // 2, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )
        cv2.putText(
            frame_copy,
            "Play Again!",
            (frame_copy.shape[1] // 2 - 100, frame_copy.shape[0] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
        cv2.imshow("YOLO - Players Detection", frame_copy)
        if cv2.waitKey(30) & 0xFF == 27:
            break


def main():
    model = YOLO('yolo11-rps-detection.pt')

    # PERSISTENT mask flags:
    # True => that player has been caught cheating and is permanently masked red.
    p1_mask = False
    p2_mask = False

    # How many wins needed
    number_of_wins = 1

    # Scores
    player1_score = 0
    player2_score = 0

    while True:
        cap = cv2.VideoCapture(0)

        # 1) PRE-PLAY PHASE
        pre_play(cap, model, p1_mask, p2_mask, number_of_wins, player1_score, player2_score)

        # 2) COUNTING / SUSPICIOUS MOVEMENT PHASE
        p1_cheated, p2_cheated = counting(cap, model, p1_mask, p2_mask,
                                          number_of_wins, player1_score, player2_score)
        if p1_cheated:
            p1_mask = True
            player1_score -= 1
        if p2_cheated:
            p2_mask = True
            player2_score -= 1
        if p1_cheated or p2_cheated:
            show_cheater_message(cap, p1_cheated, p2_cheated)
            cap.release()
            cv2.destroyAllWindows()
            continue

        # 3) FINALIZE AND CHECK DECISIONS (LOCK + POST-LOCK CHECK)
        p1_label, p2_label, p1_cheated, p2_cheated = finalize_and_check_decisions(
            cap, model, p1_mask, p2_mask,
            final_decision_lock_time=2.0,
            post_lock_check_time=2.0
        )
        if p1_cheated:
            p1_mask = True
            player1_score -= 1
        if p2_cheated:
            p2_mask = True
            player2_score -= 1

        if p1_cheated or p2_cheated:
            show_cheater_message(cap, p1_cheated, p2_cheated)
            cap.release()
            cv2.destroyAllWindows()
            continue

        # 4) DETERMINE WINNER (RPS LOGIC)
        if p1_label == "Rock" and p2_label == "Scissors":
            player1_score += 1
        elif p1_label == "Scissors" and p2_label == "Paper":
            player1_score += 1
        elif p1_label == "Paper" and p2_label == "Rock":
            player1_score += 1
        elif p2_label == "Rock" and p1_label == "Scissors":
            player2_score += 1
        elif p2_label == "Scissors" and p1_label == "Paper":
            player2_score += 1
        elif p2_label == "Paper" and p1_label == "Rock":
            player2_score += 1

        # 5) CHECK IF SOMEBODY REACHED THE WIN THRESHOLD
        if player1_score >= number_of_wins or player2_score >= number_of_wins:
            if player1_score >= number_of_wins:
                # Player 1 is winner
                show_winner_animation(cap, is_p1_winner=True, is_p2_winner=False)
            elif player2_score >= number_of_wins:
                # Player 2 is winner
                show_winner_animation(cap, is_p1_winner=False, is_p2_winner=True)
            print("Game Over!")
            cap.release()
            cv2.destroyAllWindows()
            break

        # If no winner yet, release and destroy windows, then loop again
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
