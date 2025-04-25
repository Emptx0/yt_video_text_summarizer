from audio_loader import Downloader
from audio_processor import AudioProcessor
from nlp_model import TextSummarizer

if __name__ == '__main__':
    downloader = Downloader()
    while True:
        link = input('Enter video link: ')
        if downloader(link):
            break

    processor = AudioProcessor(verbose=True)
    video_text = processor.get_video_text()

    summarizer = TextSummarizer(video_text, verbose=True)
    summarizer.get_summary()
