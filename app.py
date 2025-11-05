from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from gtts import gTTS
import requests
import os
import uvicorn

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------
# Replace this URL with your deployed CNN FastAPI endpoint on Railway
PLANT_API_URL = "https://web-production-b516.up.railway.app/predict"  

# Create static directory for audio files
os.makedirs("static", exist_ok=True)

# --------------------------------------------------------
# Initialize FastAPI
# --------------------------------------------------------
app = FastAPI(title="Voice-Enabled Plant Classifier", version="1.0.0")

# Enable CORS (for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ You can restrict this for security later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------
# Routes
# --------------------------------------------------------
@app.get("/")
def root():
    """Health check route"""
    return {"message": "Voice Plant Classifier API is running 🚀"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Receives image → Sends to Plant API → Converts prediction to speech"""
    try:
        # Read uploaded file bytes
        image_bytes = await file.read()

        # Send image to CNN model API
        files = {"file": (file.filename, image_bytes, file.content_type)}
        response = requests.post(PLANT_API_URL, files=files)

        if response.status_code != 200:
            return JSONResponse({"error": "Prediction API failed", "details": response.text}, status_code=500)

        data = response.json()
        predicted_class = data.get("predicted_class", "Unknown")
        confidence = data.get("confidence", 0.0)
        result_text = f"The plant is {predicted_class} with confidence of {confidence * 100:.2f} percent."

        # Convert text to speech
        audio_path = "static/result.mp3"
        tts = gTTS(text=result_text, lang="en")
        tts.save(audio_path)

        print("✅ Prediction:", result_text)

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "message": result_text,
            "audio_url": "/get_audio"
        }

    except Exception as e:
        print("❌ Error:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/get_audio")
def get_audio():
    """Serves the generated audio file"""
    audio_file = "static/result.mp3"
    if os.path.exists(audio_file):
        return FileResponse(audio_file, media_type="audio/mpeg")
    return JSONResponse({"error": "Audio not found"}, status_code=404)


# --------------------------------------------------------
# Entry point (for local & Railway)
# --------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Railway provides PORT env variable
    uvicorn.run("app:app", host="0.0.0.0", port=port)