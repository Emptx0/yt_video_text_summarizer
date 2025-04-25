from audio_loader import Downloader
from audio_processor import AudioProcessor
from nlp_model import TextSummarizer


if __name__ == '__main__':
    downloader = Downloader()
    link = input('Enter video link: ')
    downloader(link)

    processor = AudioProcessor()
    video_text = processor.get_video_text(verbose=True)

    summarizer = TextSummarizer(video_text, verbose=True)
    summarizer.get_summary()
