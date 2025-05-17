import os
import cv2
import cvzone
from cvzone.ClassificationModule import Classifier

# Start video capture
video = cv2.VideoCapture(0)

# Initialize classifier with the .h5 model and labels
model_path = 'Model/keras_model.h5'
labels_path = 'Model/labels.txt'
classifier = Classifier(model_path, labels_path)

# Load arrow indicator (with transparency)
arrow_img = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)

# Prepare lists for waste items and bin graphics
waste_images = []
for filename in os.listdir("Waste"):
    waste_images.append(cv2.imread(f"Waste/{filename}", cv2.IMREAD_UNCHANGED))

bin_images = []
for filename in os.listdir("Bins"):
    bin_images.append(cv2.imread(f"Bins/{filename}", cv2.IMREAD_UNCHANGED))

# Map classifier output to bin index
waste_to_bin = {
    0: 0,  # default (“no detection” will point to bin_images[0])
    1: 0, 2: 0,   # Recyclable
    3: 3, 4: 3,   # Residual
    5: 1, 6: 1,   # Hazardous
    7: 2, 8: 2    # Food
}

# Tracking accuracy (optional)
correct, total = 0, 0

# Function to overwrite current results to HTML
def write_results(html_file, waste_name, bin_name, pct):
    content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waste Classification Results</title>
    <style>
      body {{ font-family: Arial, sans-serif; background: #f4f4f4; text-align: center; }}
      .box {{ background: white; margin: 50px auto; padding: 20px; border-radius: 10px;
              box-shadow: 0 0 10px rgba(0,0,0,0.1); width: 60%; }}
      h1 {{ color: #6ca2dd; }}
      p {{ font-size: 18px; color: #333; }}
    </style>
</head>
<body>
  <div class="box">
    <h1>Waste Classification Results</h1>
    <p><strong>Waste Type:</strong> {waste_name}</p>
    <p><strong>Suggested Bin:</strong> {bin_name}</p>
    <p><strong>Accuracy:</strong> {pct:.2f}%</p>
  </div>
</body>
</html>"""
    with open(html_file, 'w') as f:
        f.write(content)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize and prepare background
    resized_cam = cv2.resize(frame, (454, 340))
    background = cv2.imread('bgimg.png')

    # Get prediction: returns [label_text, class_id, confidence]
    _, pred_id, _ = classifier.getPrediction(frame)

    # Overlay if something’s detected
    if pred_id != 0:
        # Waste image at top-right
        waste_overlay = waste_images[pred_id - 1]
        background = cvzone.overlayPNG(background, waste_overlay, (909, 127))
        # Arrow pointing to bin
        background = cvzone.overlayPNG(background, arrow_img, (978, 320))

    # Determine bin index and overlay bin graphic
    bin_index = waste_to_bin.get(pred_id, 0)
    background = cvzone.overlayPNG(background, bin_images[bin_index], (895, 374))

    # Place webcam feed in the scene
    background[148:148 + 340, 159:159 + 454] = resized_cam

    # Display final output
    cv2.imshow("Waste Sorter", background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
