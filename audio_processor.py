from transformers import pipeline, logging
import time
import warnings


class AudioProcessor:
    def __init__(self, device=-1, verbose=False):
        warnings.simplefilter("ignore")
        logging.set_verbosity_error()

        self.verbose = verbose
        self.device = device
        self.model = pipeline(task='automatic-speech-recognition',
                              model='openai/whisper-large-v3-turbo',
                              return_timestamps=True,
                              device=device)

    def get_video_text(self):
        if self.verbose:
            print('\n  - (audio processor) Starting...')
            start_time = time.time()
            text = self.model('audio.mp3')

            print('  - (audio processor) Model: openai/whisper-large-v3-turbo')
            if self.device == -1:
                print('  - (audio processor) Device set to use cpu')
            else:
                print(f'  - (audio processor) Device set to use cuda: {self.device}')
            print(f'  - (audio processor) Running for {(time.time() - start_time):.2f} seconds\n')

        else:
            text = self.model('audio.mp3')

        text = text["text"]

        return text
