from flask import Flask, request, send_file
import requests
from io import BytesIO
from PIL import Image
import pytest

# Create Flask app
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

    # Check for empty prompt
    if not text_prompt.strip():
        return "<h1>Error:</h1><p>Text prompt cannot be empty.</p>"

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


# Pytest testing code (defined within app.py)

@pytest.fixture
def client():
    """Fixture to provide a test client for the app"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    """Test the home page is accessible"""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Generate an Image from Text Prompt" in response.data

def test_generate_image_valid_prompt(client):
    """Test generating an image with a valid text prompt"""
    response = client.post('/generate_image', data={'text_prompt': 'A beautiful sunset over the mountains'})
    assert response.status_code == 200
    
    # Check if the response is an image
    image = Image.open(BytesIO(response.data))
    assert isinstance(image, Image.Image)

def test_generate_image_invalid_prompt(client):
    """Test the error handling when the API fails"""
    # Send an empty prompt or invalid data to trigger an error
    response = client.post('/generate_image', data={'text_prompt': ''})
    assert response.status_code == 200
    assert b"Error:" in response.data

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
    
    # Optionally run the tests when app starts
    pytest.main(["-v", "--maxfail=1"])  # Runs pytest automatically after the app starts
