import torch
import torchaudio

DEFAULT_TRANSFORM = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, hop_length=161, n_mels=64),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80))

class WaveformToSpectrogram:
    transform = DEFAULT_TRANSFORM

    def __init__(self, n_fft=320, n_mels=64, hop_length=161):
        transform = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels),
            torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80))

    def preprocess_waveform(waveform, length=16000, transform=transform, padLR=False):
      padding = int((length - waveform.shape[1]))
      if padLR:
          waveform = torch.roll(torch.nn.functional.pad(waveform, (0, padding)), padding // 2)
      else:
          waveform = torch.nn.functional.pad(waveform, (0, padding))

      features = transform(waveform)
      return(features)

    def preprocess_file(filePath, duration=1, transform=transform, padLR=False):
        waveform, sr = torchaudio.load(filePath)
        if sr == 16000:
            return(preprocess_waveform(waveform,int(duration*sr),transform,padLR))
        else:
            print("Error: Sample Rate must be 16000")
            return(-1)
