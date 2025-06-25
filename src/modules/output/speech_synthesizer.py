from abc import ABC, abstractmethod

class SpeechSynthesizer(ABC):
    @abstractmethod
    def speak(self, text: str):
        pass

    @abstractmethod
    def shutdown(self):
        pass


