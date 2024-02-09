import cv2
import argparse
import numpy as np
import time

import torch
from torchvision import transforms

from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import non_max_suppression_kpt,strip_optimizer
from utils.plots import output_to_keypoint, colors,plot_one_box_kpt


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

@torch.no_grad()
class PoseDetector:
    def __init__(self, poseweights="yolov7-w6-pose.pt", frame="input.png", device='cpu', 
                 view_img=False, save_conf=False, line_thickness=3, hide_labels=False, hide_conf=True):
        self.poseweights = poseweights
        self.frame = frame
        self.device = device
        self.view_img = view_img
        self.save_conf = save_conf
        self.line_thickness = line_thickness
        self.hide_labels = hide_labels
        self.hide_conf = hide_conf

    def run(self):
        device = select_device(self.device)
        half = device.type != 'cpu'

        model = attempt_load(self.poseweights, map_location=device)
        _ = model.eval()
        names = model.module.names if hasattr(model, 'module') else model.names

        orig_image = cv2.imread(self.frame)
        orig_image = cv2.resize(orig_image, (960, 960))
        height, frame_width = orig_image.shape[:2]
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        image = image.to(device)
        image = image.float()

        with torch.no_grad():
            output_data, _ = model(image)

        output_data = non_max_suppression_kpt(output_data, 0.25, 0.65, 
                                              nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)

        output = output_to_keypoint(output_data)

        im0 = image[0].permute(1, 2, 0) * 255
        im0 = im0.cpu().numpy().astype(np.uint8)
        
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

        for i, pose in enumerate(output_data):
            if len(output_data):
                for c in pose[:, 5].unique():
                    n = (pose[:, 5] == c).sum()
                    print(f"No of Objects in Current Frame : {n}")
                
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:,:6])):
                    c = int(cls)
                    kpts = pose[det_index, 6:]
                    label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True), 
                                     line_thickness=self.line_thickness, kpt_label=True, kpts=kpts, steps=3, 
                                     orig_shape=im0.shape[:2])

        cv2.imwrite('result.jpg', im0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--frame', type=str, default='input.png', help='path of image') # source
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')   #device arugments
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels') #save confidence in txt writing
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)') #box linethickness
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels') #box hidelabel
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences') #boxhideconf
    opt = parser.parse_args()
    return opt


def main(opt):
    start_time = time.time()
    detector = PoseDetector(poseweights=opt.poseweights, frame=opt.frame, device=opt.device, save_conf=opt.save_conf, 
                            line_thickness=opt.line_thickness, hide_labels=opt.hide_labels, 
                            hide_conf=opt.hide_conf)

    detector.run()
    end_time = time.time()  # End time after the function runs
    runtime = end_time - start_time  # Calculate the total runtime
    print(f"Function runtime: {runtime} seconds")

if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device,opt.poseweights)
    main(opt)