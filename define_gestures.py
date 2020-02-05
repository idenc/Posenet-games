import numpy as np


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def check_jump(state):
    if not any("RUNNING" in string for string in state.gesture):
        for frame in state.keypoints:
            l_hand_y = frame[9][0]
            r_hand_y = frame[10][0]
            l_elbow_y = frame[7][0]
            r_elbow_y = frame[8][0]
            print(l_hand_y)
            print(r_hand_y)
            print()
            if l_hand_y != 0.0 and r_hand_y != 0.0 and l_hand_y < l_elbow_y and r_hand_y < r_elbow_y:
                if "JUMP" not in state.gesture:
                    state.gesture.append("JUMP")
                state.keyboard.press('f')
                return


def check_run(state):
    if not any("RUNNING" in string for string in state.gesture) and "JUMP" not in state.gesture:
        for frame in state.keypoints:
            l_shoulder = frame[5]
            r_shoulder = frame[6]
            l_elbow = frame[7]
            r_elbow = frame[8]
            l_wrist = frame[9]
            r_wrist = frame[10]
            delta = euclidean_distance(l_shoulder, r_shoulder) * 0.5
            dir = ""
            # Check that wrist x coord is at least delta distance from shoulder x coord
            if abs(l_shoulder[1] - l_wrist[1]) > delta:  # Check left arm
                dir += "LEFT"
            if abs(r_shoulder[1] - r_wrist[1]) > delta:  # Check right arm
                dir += "RIGHT"
            if dir != "" and dir != "LEFTRIGHT":
                if dir == "LEFT":
                    if l_wrist[0] < l_elbow[0]:
                        state.jump()
                    state.keyboard.press('a')
                elif dir == "RIGHT":
                    if r_wrist[0] < r_elbow[0]:
                        state.jump()
                    state.keyboard.press('d')
                run_str = "RUNNING " + dir
                if run_str not in state.gesture:
                    state.gesture.append(run_str)
                return
