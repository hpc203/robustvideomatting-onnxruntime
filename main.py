import cv2
import onnxruntime as ort
import numpy as np
import argparse

class RobustVideoMatting():
    def __init__(self, modelpath, downsample_ratio):
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(modelpath, so)
        self.input_names = [ip.name for ip in self.net.get_inputs()]
        self.input_tensors = {ip.name: np.zeros((1,1,1,1), dtype=np.float32) for ip in self.net.get_inputs()}
        self.input_tensors[self.input_names[-1]] = np.array([downsample_ratio], dtype=np.float32)
        self.bgr = np.array([120, 255, 153], dtype=np.float32).reshape(1, 3, 1, 1)
    def detect(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32) / 255.0
        self.input_tensors[self.input_names[0]] = blob
        outs = self.net.run(None, self.input_tensors)
        com = outs[0] * 255 * outs[1] + self.bgr * (1 - outs[1])
        com = com.astype(frame.dtype).squeeze(axis=0).transpose(1,2,0)
        return cv2.cvtColor(com, cv2.COLOR_RGB2BGR)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='testdata/test.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/rvm_mobilenetv3_fp32.onnx', help='onnxmodel path')
    args = parser.parse_args()

    if not args.imgpath.endswith('.mp4'):
        net = RobustVideoMatting(args.modelpath, 0.25)
        srcimg = cv2.imread(args.imgpath)
        merge_img = net.detect(srcimg)

        cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
        cv2.imshow('srcimg', srcimg)
        winName = 'Deep learning object detection in ONNXRuntime'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.imshow(winName, merge_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        net = RobustVideoMatting(args.modelpath, 0.4)
        cap = cv2.VideoCapture(args.imgpath)
        if not cap.isOpened():
            exit("Video open failed.")
        status = True
        while status:
            status, frame = cap.read()
            if not status:
                print("Done processing !!!")
                break
            merge_img = net.detect(frame)
            cv2.namedWindow('srcimg', cv2.WINDOW_NORMAL)
            cv2.imshow('srcimg', frame)
            winName = 'Deep learning object detection in ONNXRuntime'
            cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
            cv2.imshow(winName, merge_img)
            cv2.waitKey(1)
        cv2.destroyAllWindows()