import speech_recognition
import pyttsx3



sr= speech_recognition.Recognizer()


with speech_recognition.Microphone() as source2:


    print("Silence please ")
    sr.adjust_for_ambient_noise(source2, duration =2)
    print(" Speak now please .....")
    audio2= sr.listen(source2)
    textt= sr.recognize_google(audio2)
    textt= textt.lower()
    print("Did you say : "+ textt)
    


#alternate colab and ipynb code cos the above code was not working there
import speech_recognition as sr
from pydub import AudioSegment

# Upload an audio file in Colab
from google.colab import files
uploaded = files.upload()

# Get the filename of the uploaded audio file
audio_file_name = list(uploaded.keys())[0]

# Load the audio file
audio = AudioSegment.from_file(audio_file_name)

# Convert the AudioSegment to raw audio data
audio_data = audio.raw_data

# Initialize the recognizer
sr_obj = sr.Recognizer()

# Convert the raw audio data to AudioData
audio_data = sr.AudioData(audio_data, audio.frame_rate, audio.sample_width)

# Recognize speech using Google's speech recognition
try:
    print("Transcribing audio...")
    text = sr_obj.recognize_google(audio_data)
    print("Transcription: " + text)

except sr.UnknownValueError:
    print("Google Speech Recognition could not understand the audio")

except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
