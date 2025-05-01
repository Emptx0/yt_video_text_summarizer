# YouTube video text summarizer
The text from YouTube videos is transcribed using OpenAI's Whisper model, then processed by a simple custom NLP model to generate a concise summary. \
\
![1](https://github.com/user-attachments/assets/2ca23306-ee87-43c9-9f34-3c94ba3503db) \
\
By default, Whisper runs on the CPU, but this can be changed to GPU if your graphics card supports CUDA by using `AudioProcessor(device="your CUDA device")`. \
\
**! PyTorch with CUDA support is not installed via requirements.txt !** \
\
[Go here](https://pytorch.org/get-started/locally/) to install PyTorch with CUDA 12.8 support.
### 📋 Verbose
Use `verbose=True` in `AudioProcessor()` and `TextSummarizer()` to get more information. \
\
![image](https://github.com/user-attachments/assets/6e017e1e-c01b-4a73-96e8-ad1741355786)
### ⚙️ Requirements
`pip install -r requirements.txt`
### 📓 Jupyter notebook
Prototype here: [yt_video_text_summarizer_notebook](https://github.com/Emptx0/yt_video_text_summarizer_notebook).
