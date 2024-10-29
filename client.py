import argparse
import requests
from PIL import Image
from io import BytesIO


# Set up argument parser
parser=argparse.ArgumentParser(description="Send image, video, webcam to API")
parser.add_argument('--image_path', type=str, help="Path to image")
parser.add_argument('--video_path', type=str, help="Path to video")
parser.add_argument('--camera_idx', type=int, help="Camera input (0 or 1)")

args=parser.parse_args()

# Set up API URL
url = "http://0.0.0.0:8000/predict"

# Define the parameters for the GET request
params = {}

# Check which argument is provided
if args.image_path:
    params["image_path"]=args.image_path
elif args.video_path:
    params["video_path"]=args.video_path
elif args.camera_idx is not None:
    params["camera_idx"]=args.camera_idx
else:
    print("Provide a valid choice")
    exit(1)

# Send the request
response = requests.get(url, params=params)

# Check if the response is successful
if response.status_code == 200:
    # If response is an image
    if "image/png" in response.headers.get("Content-Type", ""):
        image = Image.open(BytesIO(response.content))
        # Display the image
        image.show()
    else:
        # If response is JSON (for video_path or camera_idx)
        print(response.json())
else:
    print(f"Error: {response.status_code}")
    print(response.json())