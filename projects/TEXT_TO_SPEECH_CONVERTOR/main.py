from gtts import gTTS
import IPython.display as ipd

text = "Hello, this is a sample text to be converted to speech."
tts = gTTS(text)
tts.save("output.mp3")

# Play the generated audio
ipd.Audio("output.mp3")

