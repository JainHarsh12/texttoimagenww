from flask import Flask, request, send_file
import requests
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Replace with your Hugging Face API key
API_URL = "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
HUGGINGFACE_API_KEY = "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"

headers = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}"
}

def query_huggingface_api(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to get a response. Status code: {response.status_code}, Response: {response.text}")

@app.route('/')
def index():
    # Inline HTML for the homepage
    return '''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Image Generator</title>
        </head>
        <body>
            <h1>Generate an Image from Text Prompt</h1>
            <form action="/generate_image" method="post">
                <label for="text_prompt">Enter your text prompt:</label><br>
                <textarea id="text_prompt" name="text_prompt" rows="4" cols="50" required></textarea><br><br>
                <button type="submit">Generate Image</button>
            </form>
        </body>
        </html>
    '''

@app.route('/generate_image', methods=['POST'])
def generate_image():
    text_prompt = request.form['text_prompt']

    # Send request to Hugging Face API
    payload = {"inputs": text_prompt}
    try:
        image_bytes = query_huggingface_api(payload)
        image = Image.open(BytesIO(image_bytes))
        
        # Save the image in memory to serve it later
        img_io = BytesIO()
        image.save(img_io, 'JPEG', quality=70)
        img_io.seek(0)

        return send_file(img_io, mimetype='image/jpeg')
    except Exception as e:
        return f"<h1>Error:</h1><p>{str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True)