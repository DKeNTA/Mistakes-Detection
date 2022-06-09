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
from preprocess import MyData_loader, load_dataset_for_test
from test import TesterDeepSVDD

FRAME_SIZE = 512
HOP_LENGTH = 256
DURATION = 0.37  # in seconds
SAMPLE_RATE = 22050
N_MELS = 128
LATENT_DIM = 64

class MistakesDetection:
    def __init__(self, parameters_path):

        self.padder = Padder()
        self.extractor = LogSpectrogramExtractor(SAMPLE_RATE, FRAME_SIZE, HOP_LENGTH, N_MELS)
        self.normaliser = MinMaxNormaliser(0, 1)
        self.num_samples = int(SAMPLE_RATE * DURATION)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.deep_SVDD = TesterDeepSVDD(parameters_path, LATENT_DIM)

    def record(self, duration, save=False):
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

        print("Now Recording...")
        for i in range(0, int(sr / chunk * int(duration))):
            data = np.frombuffer(stream.read(chunk, exception_on_overflow=False), dtype='float32')
            record_data.append(data)

        record_data = list(itertools.chain.from_iterable(record_data))
        signal = np.array(record_data)
        print("Done.")

        stream.close()
        p.terminate()

        if save == True:
            now = dt.datetime.now()
            dir_name = now.strftime('%Y%m%d')
            file_name = now.strftime('%H%M%S')
            save_dir = '../datasets/record/{}'.format(dir_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file = '{}.wav'.format(file_name)
            file = os.path.join(save_dir, file)
        else:
            file = 'tmp_record.wav'

        sf.write(file, signal, sr, subtype="PCM_16")

        return signal, sr, file

    def _is_padding_necessary(self, signal):
        if len(signal) < self.num_samples:
            return True
        return False

    def _apply_padding(self, signal):
        num_missing_samples = self.num_samples - len(signal)
        padded_signal = self.padder.right_pad(signal, num_missing_samples)
        return padded_signal

    def preprocess(self, signal, sr=44100, trim=False):
        if trim:
            signal, _ = librosa.effects.trim(signal, top_db=20)
            print('trimmed')
        if sr != SAMPLE_RATE:
            signal = librosa.resample(signal, sr, SAMPLE_RATE)
        signal = signal[:self.num_samples+1]
        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        data = self.transform(norm_feature).unsqueeze(1)

        return data

    def onset_separation(self, y, sr=44100, hop_length=512):
        #y = self.normaliser.normalise(y)
        """
        rms = librosa.feature.rms(y, frame_length=1024, hop_length=hop_length, center=True)

        onset_envelope = rms[0, 1:] - rms[0, :-1]  # フレーム間の RMS の差分を計算
        onset_envelope = np.maximum(0.0, onset_envelope)  # 負数は0　　
        onset_envelope = onset_envelope / onset_envelope.max()  # 正規化

        
        pre_max = 30 / 1000 * sr // hop_length
        post_max = 0 / 1000 * sr // hop_length + 1
        pre_avg = 100 / 1000 * sr // hop_length
        post_avg = 100 / 1000 * sr // hop_length + 1
        wait = 65 / 1000 * sr // hop_length
        
        w = 175 / 1000 * sr // hop_length
        delta = 0.12
        onset_frames = librosa.util.peak_pick(onset_envelope, w, w, w, w, delta, w)  # オンセット検出
        #onset_backtrack = librosa.onset.onset_backtrack(onset_frames, onset_envelope)  # オンセット前の検出
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        """
        """
        is_onset_mistakes = True
        while is_onset_mistakes == True:
            is_onset_mistakes = False
            for i in range(1, len(onset_times)):
                if onset_times[i]-onset_times[i-1] <= 0.25:
                    onset_times = np.delete(onset_times, i)
                    is_onset_mistakes = True
                    break
        """
        w = 175 / 1000 * sr // hop_length
        d = {'pre_max': w, 'post_max': w, 'pre_avg': w, 'post_avg': w, 'wait': w, 'delta': 0.12}
        onset_times = librosa.onset.onset_detect(y, sr, hop_length=1024, normalize=True, units='time', **d)

        onset_samples = onset_times * sr
        y_onset_separation = []
        for i in range(len(onset_samples)):
            if i == len(onset_samples)-1:
                tmp = y[int(onset_samples[i]):]
            else:
                tmp = y[int(onset_samples[i]):int(onset_samples[i+1])]

            y_onset_separation.append(tmp)

        return y_onset_separation, onset_times

    def random_load(self, dir, offset):
        filename = random.choice([x for x in os.listdir(dir) if os.path.isfile(os.path.join(dir, x))])
        file = os.path.join(dir, filename)
        print('loading {}'.format(file))
        y, sr = librosa.load(file, sr=None, offset=offset)
        return y, sr, file

    def all_load(self, dir, sr, offset=None):
        signals = []
        filenames = []
        for root, _, file_names in os.walk(dir):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                signal, _ = librosa.load(file_path, sr=sr, offset=offset)
                signals.append(signal)
                filenames.append(file_name)

        return signals, filenames

    def melspectrograms_load(self, dir):
        spectrograms = []
        filename = []
        for root, _, file_names in os.walk(dir):
            for file_name in file_names:
                file_path = os.path.join(root, file_name)
                spectrogram = np.load(file_path)
                spectrograms.append(spectrogram)
                filename.append(file_name)

        return spectrograms, filename

    def mistakes_detection(self, file_names, scores, th):
        for file_name, score in zip(file_names, scores):
            if score >= float(th):
                print('{}: {:.5f}'.format(file_name, score))

    def play_wav(self, file):
        cmd = 'afplay {}'.format(file)
        subprocess.Popen(cmd, shell=True)

    def plot(self, y, sr, file, scores=[], onsets=[], mistakes=[], th=None):
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(2, 1, 1)
        y /= np.abs(y).max()
        librosa.display.waveplot(y, sr, ax=ax1)
        ax1.set(title='Wave')
        ax1.label_outer()
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
            #offsets = onsets + 0.37
            plt.vlines(onsets, -1, 1, color='y', linestyle='--')
            #plt.vlines(offsets, -1, 1, color='m', linestyle='--')
            #plt.vlines(mistakes, -1, 1, color='r', linestyle='--')


        #ax2 = fig.add_subplot(3, 1, 2, sharex=ax1)
        #librosa.display.waveplot(y, sr, x_axis='time')


        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        if sr != SAMPLE_RATE:
            y = librosa.resample(y, sr, SAMPLE_RATE)
        feature = self.extractor.extract(y)
        norm_feature = self.normaliser.normalise(feature)
        librosa.display.specshow(norm_feature, sr=44100, x_axis='time', y_axis='mel', ax=ax2)
        ax2.set_title('Mel Spectrogram')
        #fig.colorbar(img, ax=ax2, format="%+2.f dB")

        """
        ax3 = fig.add_subplot(3, 1, 3, sharex=ax1)
        plt.bar(onsets, scores, width=0.15, align='edge', linewidth=5, color='r')
        plt.ylim(0,3)
        plt.hlines(th, 0, len(y)/sr*2, color='black', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Score')
        """

        plt.tight_layout()
        self.play_wav(file)
        plt.show()


    def save_result(self, i, signal, sr):
        now = dt.datetime.now()
        time = now.strftime('%Y%m%d-%H%M%S')
        save_dir = 'result/{}'.format(time)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file = '{}/mistakes-{}.wav'.format(save_dir, i)
        sf.write(file, signal, sr, subtype="PCM_16")

    def run(self, args):

        if args.record:
            y, sr, file = self.record(args.duration, args.record_save)
        elif args.wavfile != None:
            ext = os.path.splitext(args.wavfile)[1]
            if ext == '.wav':
                file = args.wavfile
                y, sr = librosa.load(file, sr=44100, offset=args.offset)
                #y = librosa.effects.time_stretch(y, args.rate)
            elif ext == '':
                y, sr, file = self.random_load(args.wavfile, args.offset)
            else:
                print('file Error')
                return
        elif args.directory != None:
            signals, file_names = self.all_load(args.directory, sr=44100, offset=args.offset)
            file_num = len(file_names)
            mistakes_num = 0
            total_score = 0
            for signal, file_name in zip(signals, file_names):
                  data = self.preprocess(signal, sr=44100, trim=args.trim)
                  score = self.deep_SVDD.test_data(data)
                  if score >= args.threshold and args.threshold > 0:
                      print('{}: {:.5f}'.format(file_name, score))
                      mistakes_num += 1
                  elif score <= -args.threshold and args.threshold < 0:
                      print('{}: {:.5f}'.format(file_name, score))
                      mistakes_num += 1
                  total_score += score
            detection_rate = mistakes_num / file_num * 100
            mean_score = total_score / len(signals)
            print(f"\nMean Score : {mean_score} \nDetection Rate : {detection_rate:.3f} %")
            return

        elif args.melspectrograms_dir != None:
            melspectrograms, file_names = self.melspectrograms_load(args.melspectrograms_dir)
            file_num = len(file_names)
            mistakes_num = 0
            total_score = 0
            for spectrogram, file_name in zip(melspectrograms, file_names):
                  data = self.transform(spectrogram).unsqueeze(1)
                  score = self.deep_SVDD.test_data(data)
                  if score >= args.threshold and args.threshold > 0:
                      print('{}: {:.5f}'.format(file_name, score))
                      mistakes_num += 1
                  elif score <= -args.threshold and args.threshold < 0:
                      print('{}: {:.5f}'.format(file_name, score))
                      mistakes_num += 1
                  total_score += score
            detection_rate = mistakes_num / file_num * 100
            mean_score = total_score / len(melspectrograms)
            print(f"\nMean Score : {mean_score} \nDetection Rate : {detection_rate:.3f} %")
            return

        elif args.test == True:
            dataset, labels = load_dataset_for_test(args.dataset)
            test_dataset = MyData_loader(dataset, labels, self.transform)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            self.deep_SVDD.test_dataset(test_loader)
            #self.deep_SVDD.test_dataset_PN(test_loader, args.threshold)
            return
        """
        else:
            dataset, file_names = load_dataset(args.dataset)
            dataset_ = MyData_loader(dataset, self.transform)
            dataloader = DataLoader(dataset_, batch_size=64, shuffle=False)
            scores = self.deep_SVDD.test_dataset(dataloader)
            self.mistakes_detection(file_names, scores, args.threshold)
            return
        """

        if args.onset:
            signals, onset_times = self.onset_separation(y, sr=sr)
            scores = []
            mistakes = []
            for i, signal in enumerate(signals):
                  data = self.preprocess(signal, args, sr, trim=args.trim)
                  score = self.deep_SVDD.test_data(data)
                  scores.append(score)
                  print('{} score: {:.5f}'.format(i, score))
                  if score > args.threshold:
                    mistakes.append(i)
                    if args.result:
                        self.save_result(i, signal, sr)
            self.plot(y, sr, file, scores=scores, onsets=onset_times, mistakes=mistakes, th=args.threshold)

        else:
            data = self.preprocess(y, sr=sr, trim=args.trim)
            score = self.deep_SVDD.test_data(data)
            print('score: {:.5f}'.format(score))
            self.plot(y, sr, file, scores=score)

        print("Finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-d', '--data', choices=['record', 'dir', 'wav', 'mel', 'dataset'], default='dataset')
    parser.add_argument('-prm', '--parameters_path', default='../../Downloads/network_parameters.pth')
    parser.add_argument('-rec', '--record', action='store_true')
    parser.add_argument('--record_save', action='store_true')
    parser.add_argument('-d', '--duration', default=5)
    parser.add_argument('-dir', '--directory')
    parser.add_argument('-spec', '--melspectrograms_dir')
    parser.add_argument('-wf', '--wavfile')
    parser.add_argument('--dataset', default='../datasets/test/melspectrograms')
    parser.add_argument('--trim', action='store_true')
    parser.add_argument('-th', '--threshold', type=float, default=1.0)
    parser.add_argument('--rate', type=float, default=1)
    parser.add_argument('--offset', type=float)
    parser.add_argument('-onset-off', '--onset', action='store_false')
    parser.add_argument('--result', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    mistakesDetection = mistakesDetection(args.parameters_path)
    mistakesDetection.run(args)
