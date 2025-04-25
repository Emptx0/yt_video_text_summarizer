from pytubefix import YouTube
import os


class Downloader:
    def __call__(self, link):
        try:
            yt = YouTube(str(link))
            audio = yt.streams.filter(only_audio=True).first()
            audio.download(filename='audio.mp3')
            return True
        except:
            print('Video not found!')

    def __del__(self):
        os.remove('audio.mp3')
