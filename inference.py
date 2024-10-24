import cv2, torch
import face_recognition
import numpy as np
from torchvision import transforms
from model import vgg16
from PIL import Image
from tqdm import tqdm
import argparse  # Import argparse for handling command-line arguments


class EmotionPredictor:
    def __init__(self,model_path):
        self.model_path=model_path

        self.emotions = {
            0: ['Angry', (0,0,255), (255,255,255)],
            1: ['Disgust', (0,102,0), (255,255,255)],
            2: ['Fear', (255,255,153), (0,51,51)],
            3: ['Happy', (153,0,153), (255,255,255)],
            4: ['Sad', (255,0,0), (255,255,255)],
            5: ['Surprise', (0,255,0), (255,255,255)],
            6: ['Neutral', (160,160,160), (255,255,255)]
        }
        self.model=vgg16.load_from_checkpoint(model_path)
        print("Load model successfully!")
        
        self.model.eval()
        self.model.freeze()
        self.model.to("cpu")  # Move the model to the appropriate device (MPS or CPU)

        self.cut_size=44
        # set up transform
        self.transform=transforms.Compose([transforms.TenCrop(self.cut_size),
                                           transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                           ])
    
    def inference_image(self,image):
        # image = cv2 image array
        locations=face_recognition.face_locations(image)
        for (top,right,bottom,left) in locations:
            # prediction on face
            face=image[top:bottom, left:right,:]            # [W,H,3]
            pred=self.prediction(face)

            # draw retangular and put text
            cv2.rectangle(image,(left,top),(right,bottom), self.emotions[pred][1],2)
            cv2.rectangle(image,(left,top-30), (right,top), self.emotions[pred][1],-1)
            cv2.putText(image,
                        f'{self.emotions[pred][0]}', (left,top-5),0,0.7, self.emotions[pred][2],2)
        return image 
        
    def inference_video(self,video_path):
        cap=cv2.VideoCapture(video_path)
        width,height=int(cap.get(3)), int(cap.get(4))
        fps=cap.get(cv2.CAP_PROP_FPS)
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {num_frames} frames | Resolution: {width}x{height}")

        # codec to write new video
        out=cv2.VideoWriter(video_path[:-4]+"_out.mp4",
                        cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
        
        with tqdm(total=num_frames, desc="Processing video frames") as pbar:
            # read frame, make inference and write into a video
            while cap.isOpened():
                success,frame=cap.read()

                if not success:
                    print("Failed to read frame!")
                    break
                # process frame
                frame=self.inference_image(frame)
                out.write(frame)
                pbar.update(1)
               
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def inference_web(self,camera_idx):
        # load webcam
        cap=cv2.VideoCapture(camera_idx)

        process_this_frame=True # only process every 2nd frame for faster inference

        while True:
            ret, frame=cap.read()
            if not ret:
                print("Error: Failed to read frame.")
                break

            if process_this_frame:
                # resize 1/4 of original frame for faster inference
                small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
                face_locations=face_recognition.face_locations(small_frame)

            process_this_frame=not process_this_frame

            # give predictions to facial regions and display results
            for (top,right,bottom,left) in face_locations:

                face=small_frame[top:bottom,left:right,:]
                pred=self.prediction(face)

                # scale back up face locations to draw boxes
                top, right, bottom, left=top*4, right*4, bottom*4, left*4

                # draw rectangle and put texts around the faces
                cv2.rectangle(frame,(left,top), (right,bottom),self.emotions[pred][1],2)
                cv2.rectangle(frame,(left,top-50), (right,top),self.emotions[pred][1],-1)
                cv2.putText(frame,
                            f'{self.emotions[pred][0]}', 
                            (left,top-5), 0, 1.5,self.emotions[pred][2],2)

            # show frame
            cv2.imshow("Frame",frame)

            # set up an escape key
            if cv2.waitKey(1) & 0xFF == 27:
                break
        
    def prediction(self,face): # face = face image
        # convert to gray-scale and resize it to (48,48)
        gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,(48,48)).astype(np.uint8)

        # randomly crop the face image into 10 subimages
        inputs=self.transform(Image.fromarray(gray))      #[10,1,44,44]

        pred=self.model(inputs)                           # [10,7]
        pred=pred.mean(axis=0)                            # [7,]
        return pred.argmax().item()

if __name__=="__main__":
    """ To run: python inference.py --image_path path_to_image
    """
    # load predictor class
    predictor=EmotionPredictor("./models/best_checkpoint.ckpt")

    # set up argument parser
    parser=argparse.ArgumentParser(description="Emotion Predictor")
    parser.add_argument('--image_path', type=str, help="Path to the image for inference")
    parser.add_argument('--video_path', type=str, help="Path to the video for inference")
    parser.add_argument('--camera_idx', type=int, default=0, help="Webcam index for webcam inference")
    
    args = parser.parse_args()

    if args.image_path:
        image = cv2.imread(args.image_path)
        result = predictor.inference_image(image)
        cv2.imshow("Result", result)
        # close window when pressing esc
        while True:
            if cv2.waitKey(1) & 0xFF==27:
                break
        cv2.waitKey(1)

    elif args.video_path:
        predictor.inference_video(args.video_path)

    elif args.camera_idx is not None:
         predictor.inference_web(args.camera_idx)  
          
    else:
        print("Please enter a valid option!")
    # image=cv2.imread("data/digipen.jpeg")
    # image=predictor.inference_image(image)
    # cv2.imwrite("data/inference.png",image)
    # video_path="data/muskma.mp4"
    # predictor.inference_video(video_path)
    #predictor.inference_web(camera_idx=0)