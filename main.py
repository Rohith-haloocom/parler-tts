import os
import torch
import asyncio
from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
from datetime import datetime
import time
import numpy as np

# Create output directory if it doesn't exist
OUTPUT_DIR = 'generated_audio'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ParlerTTSService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ParlerTTSService, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    def _initialize_model(self):
        """Initialize the TTS model only once"""
        # Determine device
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load model and tokenizers
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        self.description_tokenizer = AutoTokenizer.from_pretrained(self.model.config.text_encoder._name_or_path)
        
        print("Model initialized successfully")

    async def generate_speech(self, prompt, description):
        """
        Generate speech from text prompt and description asynchronously
        
        :param prompt: Text to convert to speech
        :param description: Voice characteristics description
        :return: Path to generated audio file
        """
        start_time = time.time()

        # Tokenize inputs
        input_ids = self.description_tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        prompt_input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        # Generate speech asynchronously
        generation = await asyncio.to_thread(self.model.generate, input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()

        # Convert to supported dtype
        audio_arr = audio_arr.astype(np.float32)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"tts_output_{timestamp}.wav")

        # Save audio file asynchronously
        await asyncio.to_thread(sf.write, filename, audio_arr, self.model.config.sampling_rate)

        # Print generation time
        end_time = time.time()
        print(f"Speech generation time: {end_time - start_time:.2f} seconds")

        return filename

# Flask Application
app = Flask(__name__)
CORS(app)
tts_service = ParlerTTSService()

@app.route('/generate-speech', methods=['POST'])
async def generate_speech():
    start_time = time.time()
    try:
        # Get request data
        data = request.json
        prompt = data.get('prompt')
        description = data.get('description', 'Default female voice')

        # Validate inputs
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Generate speech asynchronously
        file_path = await tts_service.generate_speech(prompt, description)

        # Print API response time
        end_time = time.time()
        print(f"Total API response time: {end_time - start_time:.2f} seconds")

        return jsonify({
            "message": "Speech generated successfully",
            "file_path": file_path
        })

    except Exception as e:
        print(f"API request failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
async def download_file(filename):
    try:
        file_path = os.path.join(OUTPUT_DIR, filename)
        return await asyncio.to_thread(send_file, file_path, as_attachment=True)
    except Exception as e:
        print(f"File download failed: {e}")
        return jsonify({"error": "File not found"}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8400)
