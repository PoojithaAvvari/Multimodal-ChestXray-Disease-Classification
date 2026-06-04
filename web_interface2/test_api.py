import requests
import os

backend_url = 'http://127.0.0.1:8000/analyze'
image_path = os.path.join('..', 'archive', 'images', 'images_normalized', '64_IM-2218-4004.dcm.png') # Real image

print(f"Testing with image: {image_path}")

report_text = "The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. There is no pneumothorax or pleural effusion."

with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    data = {'report': report_text}
    print("Sending POST request to /analyze...")
    try:
        response = requests.post(backend_url, files=files, data=data)
        print("Status code:", response.status_code)
        with open("response.txt", "w") as rf:
            rf.write(response.text)
    except Exception as e:
        print("Error during request:", e)
        print("Error during request:", e)
        if hasattr(e, 'response') and e.response:
            print("Response:", e.response.text)
