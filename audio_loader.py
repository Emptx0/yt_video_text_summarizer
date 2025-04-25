from pytubefix import YouTube


class Downloader:
    def __call__(self, link):
        try:
            yt = YouTube(str(link))
            audio = yt.streams.filter(only_audio=True).first()
            audio.download(output_path='audio', filename='audio.mp3')
        except:
            print('No video found')
