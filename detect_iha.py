import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (check_img_size, check_requirements, check_imshow, non_max_suppression,
                           apply_classifier, scale_coords, xyxy2xywh, strip_optimizer,
                           set_logging, increment_path)
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = (opt.source, opt.weights, opt.view_img,
                                                        opt.save_txt, opt.img_size, not opt.no_trace)
    save_img = not opt.save_img and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # QR Kodlar için klasör oluşturma
    qr_dir = save_dir / 'qr_codes'
    qr_dir.mkdir(parents=True, exist_ok=True)
    qr_txt_path = qr_dir / 'qr_messages.txt'  # QR mesajlarının yazılacağı dosya

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    if(device.type != 'cpu'):
        compute_capability = torch.cuda.get_device_capability(device=device)
        half = (device.type != 'cpu') and (compute_capability[0] >= 8)  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)

        cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam', 1600, 900)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Initialize QR Code Detector
    qr_detector = cv2.QRCodeDetector()

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2]
                                     or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                   agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path_det = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image'
                             else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            # Define the target area (centered rectangle)
            h, w, _ = im0.shape
            margin_left = w * 0.25
            margin_right = w * 0.75
            margin_top = h * 0.1
            margin_bottom = h * 0.9

            # Draw the target area rectangle
            cv2.rectangle(im0, (int(margin_left), int(margin_top)),
                          (int(margin_right), int(margin_bottom)), (0, 255, 0), 3)

            # --- Başlangıç: Merkez yuvarlak ve çizgiler ---
            # Hesapla görüntünün merkez noktası
            center_x, center_y = w // 2, h // 2

            # Merkezde küçük bir yuvarlak çiz
            cv2.circle(im0, (center_x, center_y), radius=4, color=(0, 255, 0), thickness=2)  # Mavi dolu daire

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    # Bounding box boyutu kontrolü (%5 kısıtı)
                    box_w = xyxy[2] - xyxy[0]
                    box_h = xyxy[3] - xyxy[1]
                    if box_w < w * 0.05 or box_h < h * 0.05:
                        continue

                    # Kontrollü olarak hedef vuruş alanında olup olmadığını kontrol etme
                    # Nesnenin tamamının hedef alanın içinde olup olmadığını kontrol etme
                    if not (margin_left < xyxy[0] < margin_right and
                            margin_left < xyxy[2] < margin_right and  # Sağ alt köşe kontrolü
                            margin_top < xyxy[1] < margin_bottom and
                            margin_top < xyxy[3] < margin_bottom):  # Sağ üst köşe kontrolü
                        continue

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path_det + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=(0, 0, 255), line_thickness=2)  # Set color to red (BGR: (0, 0, 255))

                    # --- Çizgi çizme: Merkezden nesneye ---
                    # Nesnenin merkezini hesapla
                    obj_center_x = int((xyxy[0] + xyxy[2]) / 2)
                    obj_center_y = int((xyxy[1] + xyxy[3]) / 2)

                    # Merkezden nesnenin merkezine çizgi çiz
                    cv2.line(im0, (center_x, center_y), (obj_center_x, obj_center_y),
                             color=(0, 255, 255), thickness=2)  # Cyan renkli çizgi
            # --- Bitiş: Merkez yuvarlak ve çizgiler ---

            # --- QR Kod Okuma Başlangıç ---
            # QR kodu tespit et ve oku
            qr_data, bbox, _ = qr_detector.detectAndDecode(im0)
            if bbox is not None and qr_data:
                # Eğer QR kod tespit edildiyse, veriyi yaz
                with open(qr_txt_path, 'a') as qr_file:
                    qr_file.write(f'Frame {frame}: {qr_data}\n')

                # QR kodun etrafına kutu çiz
                bbox = bbox.astype(int).reshape(-1, 2)
                for j in range(len(bbox)):
                    cv2.line(im0, tuple(bbox[j]), tuple(bbox[(j + 1) % len(bbox)]), (0, 255, 0), 2)
                # QR kodun merkezine işaretçi ekle
                qr_center_x = int((bbox[:,0].min() + bbox[:,0].max()) / 2)
                qr_center_y = int((bbox[:,1].min() + bbox[:,1].max()) / 2)
            # --- QR Kod Okuma Bitiş ---

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow('Webcam', im0)
                if cv2.waitKey(1) == ord('q'):  # 'q' tuşu ile çıkış
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
                print(f" The video with the result is saved in: {save_path}")

    if save_txt or save_img:
        print(f"Results saved to {save_dir}")
    print(f'Done. ({time.time() - t0:.3f}s)')

# Command-line interface
if __name__ == '__main__':
    import argparse

    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5s.pt', help='model.pt path')
        parser.add_argument('--source', type=str, default='data/images', help='source')
        parser.add_argument('--img-size', type=int, default=640, help='image size')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
        parser.add_argument('--save-txt', action='store_true', help='save results to txt')
        parser.add_argument('--save-img', action='store_true', help='save results to image file')
        parser.add_argument('--view-img', action='store_true', help='show results in real time')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--no-trace', action='store_true', help='don\'t trace model')

        # --classes parametresini ekledik
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')

        # --agnostic-nms parametresini ekleyin
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')

        # Diğer parametreler
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', type=str, default='exp', help='name of the experiment')

        return parser.parse_args()

    opt = parse_opt()

    detect()