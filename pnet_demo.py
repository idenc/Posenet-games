import argparse
import time

import cv2

from posenet_interface import posenetInterface

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=50)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=480)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    pnet = posenetInterface(257)

    total_frames = 0
    time_begin = time.time()
    while 1:
        infer_begin = time.time()
        overlay_image, _ = pnet.get_image(3)

        infer_time = time.time() - infer_begin
        print("Draw time: " + str(infer_time))
        fps = 1.0 / infer_time

        cv2.putText(overlay_image, '%.2f fps' % fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0))
        cv2.imshow('posenet', cv2.cvtColor(overlay_image, cv2.COLOR_RGB2BGR))
        total_frames += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', total_frames / (time.time() - time_begin))


if __name__ == "__main__":
    main()
