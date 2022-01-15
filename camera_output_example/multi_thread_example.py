import argparse
import cv2

from time import time
from picamerathread import PiVideoStream


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='multi_thread_example.py', description='Arguments for work with raspi camera')
    parser.add_argument('-t', '--time', type=float, default=None, help='Working time in seconds (default: 60)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Output images folder (default: None)')
    parser.add_argument('--screen_output', default=False, action='store_true', help='Output images on screen (default: False)')
    parser.add_argument('--silent', default=False, action='store_true', help='Without output fps (default=False)')
    parser.add_argument('--resolution', type=int, default=640, help='Output resolution. Write 1080, 720 or 640 (default=640)')
    args = parser.parse_args()

    if args.resolution == 240:
        resolution = (432, 240)
    elif args.resolution == 360:
        resolution = (640, 368)
    elif args.resolution == 480:
        resolution = (864, 480)
    elif args.resolution == 720:
        resolution = (1280, 720)
    elif args.resolution == 1080:
        resolution = (1920, 1080)
    else:
        resolution = (320, 240)

    if args.time:
        print('Running camera for {} seconds'.format(args.time))

        frame_grabber = PiVideoStream(resolution).start()
        num_frames = 0
        past_frame_index = -1
        current_time = time()

        while time() - current_time < args.time:
            frame, next_frame_index = frame_grabber.read()
            
            if not args.silent:
                if next_frame_index != past_frame_index:
                    num_frames += 1
                    past_frame_index = next_frame_index
            
            if args.screen_output:
                try:
                    cv2.imshow('Pi Cam', frame)
                    key = cv2.waitKey(1) & 0xFF

                except cv2.error:
                    pass

            if args.output:
                try:
                    cv2.imwrite('~/Desktop/raspberry_cameras/image_examples/{1}.jpg'.format(num_frames), frame)
                except cv2.error:
                    pass
        
        frame_grabber.stop()
        print('FPS(maybe not true) :  ', num_frames / (time() - current_time))
        cv2.destroyAllWindows()