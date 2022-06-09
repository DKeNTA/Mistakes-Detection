import pyaudio
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random
import argparse
import os
import datetime as dt
import subprocess
import soundfile as sf
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms

from preprocess_signal import Padder, LogSpectrogramExtractor, MinMaxNormaliser
from test import TesterDeepSVDD

FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 0.37  # in seconds
SAMPLE_RATE = 22050
N_MELS = 128
LATENT_DIM = 64

import tkinter as tk
from tkinter import ttk
from mistakes_detection import MistakesDetection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import sounddevice as sd

MistakesDetection = MistakesDetection('weights/eta-15.pth')
boo = False


def plot(fig, y, sr, scores, onsets, mistakes, th):
    ax1 = fig.add_subplot(3, 1, 1)
    y /= np.abs(y).max()
    librosa.display.waveplot(y, sr, ax=ax1)
    ax1.set_title('Wave')
    if len(onsets) != 0:
        if len(mistakes) != 0:
            y_mistakes = np.zeros(len(y))
            for i in mistakes:
                mistakes_onset = int(onsets[i] * sr)
                if len(onsets) == i+1:
                    y_mistakes[mistakes_onset:] += y[mistakes_onset:]
                else:
                    mistakes_offset = int(onsets[i+1] * sr) - 1
                    y_mistakes[mistakes_onset:mistakes_offset] += y[mistakes_onset:mistakes_offset]
            librosa.display.waveplot(y_mistakes, sr, color='r')
        plt.vlines(onsets, -1, 1, color='y', linestyle='--')
    ax1.label_outer()

    ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, sr, SAMPLE_RATE)
    feature = MistakesDetection.extractor.extract(y)
    norm_feature = MistakesDetection.normaliser.normalise(feature)
    librosa.display.specshow(norm_feature, sr=44100, x_axis='time', y_axis='mel', ax=ax2)
    ax2.set_title('Mel Spectrogram')
    ax2.label_outer()
    #fig.colorbar(img, ax=ax2, format="%+2.f dB")

    ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
    plt.bar(onsets, scores, width=0.15, align='edge', linewidth=5, color='r')
    #onsets_ = np.round(onsets, 1).tolist()
    #plt.xticks(onsets_, rotation=45)
    plt.xticks(onsets, np.round(onsets, 1).tolist(), rotation=45)
    """
    # 棒グラフ内に数値を書く
    for onset, score in zip(onsets, scores):
        if score > th:
            s = round(float(score), 2)
            ax3.text(onset, th+0.1, s, ha='center', va='bottom')
    """
    plt.ylim(0,3)
    plt.hlines(th, 0, len(y)/sr*2, color='black', linestyle='--')
    plt.xlabel('Times')
    plt.ylabel('Score')

    plt.tight_layout()

    return fig

def mistakes_detection(y, sr, th, fig):

    signals, onsets = MistakesDetection.onset_separation(y, sr=sr)
    scores = []
    mistakes = []
    for i, signal in enumerate(signals):
          data = MistakesDetection.preprocess(signal, sr)
          score = MistakesDetection.deep_SVDD.test_data(data)
          scores.append(score)
          #print('{} score: {:.5f}'.format(i, score))
          if score > th:
            mistakes.append(i)
            #MistakesDetection.save_result(i, signal, sr)

    plot(fig, y, sr, scores, onsets, mistakes, th)

    return signals, np.round(onsets, 1).tolist()

def record(duration):
    # マイクインプット設定
    p = pyaudio.PyAudio()
    format = pyaudio.paFloat32
    sr = 44100 # サンプリング周波数
    ch = 1
    chunk = 1024  # 1度に読み取る音声のデータ幅
    stream = p.open(format = format,
                             #input_device_index = 0,
                             channels = ch,
                             rate = sr,
                             input = True,
                             #output = False,
                             frames_per_buffer = chunk)
    record_data = []

    for i in range(0, int(sr / chunk * int(duration))):
        data = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype='float32')
        record_data.append(data)

    record_data = list(itertools.chain.from_iterable(record_data))
    signal = np.array(record_data)

    stream.close()
    p.terminate()

    return signal, sr

def play_signal(signal, sr):
    sd.play(signal, sr)

def stop_playing():
    sd.stop()

def run_command(fig, canvas, frame, wavfile=None, duration=None, th=0.7):
    global boo
    if boo:
        children = frame.winfo_children()
        for child in children:
            child.destroy()

    fig.clf()

    if wavfile != None:
        y, sr = librosa.load(wavfile, sr=44100)
    else:
        y, sr = record(duration)

    signals, onsets = mistakes_detection(y, sr, th, fig)
    canvas.draw()

    play_button = tk.Button(frame, text='最初から再生',
                            command=lambda:play_signal(y, sr))

    onset_combobox = ttk.Combobox(frame, values=onsets)
    onset_play_button = tk.Button(frame, text='指定位置を再生',
                                  command=lambda:play_signal(signals[onsets.index(float(onset_combobox.get()))], sr))

    stop_button = tk.Button(frame, text='停止',
                            command=stop_playing)

    play_button.pack(pady=15)
    onset_combobox.pack()
    onset_play_button.pack()
    stop_button.pack(pady=15)

    boo = True

def open_file_command(edit_box):
    fTyp = [("","*.wav")]

    iDir = os.path.abspath(os.path.dirname(__file__))
    #tk.messagebox.showinfo('○×プログラム','処理ファイルを選択してください！')
    file = tk.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)

    edit_box.delete(0, tk.END)
    edit_box.insert(tk.END, file)

def close_window(root_frame):
    #p.terminate()
    root_frame.destroy()

# ボタンエリアのフレームを作成して返却する関数
def set_frame(root_frame):
    fig_frame = tk.Frame(root_frame)
    button_frame = tk.Frame(root_frame)
    button_frame_2 = tk.Frame(root_frame)


    fig = plt.figure(figsize=(12, 10))
    canvas = FigureCanvasTkAgg(fig, fig_frame)
    #ツールバーを表示
    toolbar=NavigationToolbar2Tk(canvas, fig_frame)
    canvas.get_tk_widget().pack()

    # テキストボックスの作成と配置
    button_frame.edit_box = tk.Entry(button_frame, width=30)

    # ボタンの作成と配置
    file_button = tk.Button(button_frame, text='オーディオファイルを選択',
                            command=lambda:open_file_command(button_frame.edit_box))

    run_button = tk.Button(button_frame, text='実行',
                           command=lambda:run_command(fig, canvas, button_frame_2, wavfile=button_frame.edit_box.get()))

    rec_duration_box = tk.Spinbox(button_frame, from_=5, to=60, increment=5)
    rec_button = tk.Button(button_frame, text='録音して実行',
                           command=lambda:run_command(fig, canvas, button_frame_2, duration=rec_duration_box.get()))

    close_button = tk.Button(root_frame, text='閉じる',
                             command=lambda:close_window(root_frame))

    file_button.pack()
    button_frame.edit_box.pack()
    run_button.pack(pady=15)
    rec_duration_box.pack(pady=10)
    rec_button.pack()
    close_button.pack(side=tk.BOTTOM)

    fig_frame.pack(side=tk.LEFT)
    button_frame.pack(pady=100)
    button_frame_2.pack()

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('1400x1000')
    root.title('Mistakes Detection')
    set_frame(root)
    root.mainloop()
