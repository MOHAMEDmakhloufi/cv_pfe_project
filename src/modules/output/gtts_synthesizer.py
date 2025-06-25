from gtts import gTTS
from io import BytesIO
import pygame
import time
import threading
import queue
from src.modules.output.speech_synthesizer import SpeechSynthesizer

class GTTSSpeechSynthesizer(SpeechSynthesizer):
    def __init__(self):
        pygame.init()
        pygame.mixer.init()
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker)
        self.speech_thread.start()

    def _wait_for_sound_end(self):
        while pygame.mixer.get_busy():
            time.sleep(0.1)

    def speak(self, text: str):
        if len(text) > 0:
            self.speech_queue.put(text)

    def _text_to_speech_internal(self, text, language='en'):
        try:
            mp3_fo = BytesIO()
            tts = gTTS(text, lang=language)
            tts.write_to_fp(mp3_fo)
            mp3_fo.seek(0)
            sound = pygame.mixer.Sound(mp3_fo)
            sound.play()
            self._wait_for_sound_end()
        except Exception as e:
            print(f"Error during text-to-speech generation: {e}")

    def _speech_worker(self):
        while True:
            sentence = self.speech_queue.get()
            if sentence is None:
                break
            self._text_to_speech_internal(sentence)

    def shutdown(self):
        self.speech_queue.put(None)
        self.speech_thread.join()
        pygame.mixer.quit()
        pygame.quit()


