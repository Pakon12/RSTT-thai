import pyaudio
import audioop
import torch
import struct
import soundfile as sf
import io
import sys
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor ,TrainingArguments

from pythainlp import correct

dev = "cpu"
if torch.cuda.is_available():  
  dev = "cuda" 
  print('-------------------------------------------gup run')
else:  
  dev = "cpu"  
  print('-------------------------------------------cup run')

def write_header(_bytes, _nchannels, _sampwidth, _framerate):
    WAVE_FORMAT_PCM = 0x0001
    initlength = len(_bytes)
    bytes_to_add = b'RIFF'
    
    _nframes = initlength // (_nchannels * _sampwidth)
    _datalength = _nframes * _nchannels * _sampwidth

    bytes_to_add += struct.pack('<L4s4sLHHLLHH4s',
        36 + _datalength, b'WAVE', b'fmt ', 16,
        WAVE_FORMAT_PCM, _nchannels, _framerate,
        _nchannels * _framerate * _sampwidth,
        _nchannels * _sampwidth,
        _sampwidth * 8, b'data')

    bytes_to_add += struct.pack('<L', _datalength)

    return bytes_to_add + _bytes

# Load pretrained processor and model
processor = Wav2Vec2Processor.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")
model = Wav2Vec2ForCTC.from_pretrained("airesearch/wav2vec2-large-xlsr-53-th")

# กำหนดค่าตัวแปรสำหรับการบันทึกเสียง
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16_000
CHUNK = 8192
SILENT_THRESHOLD = 1000
MAX_SILENCE_SECONDS = 1


def word_correction(sentence):
    newText = ""
    for subword in sentence.split(" "):
        if len(newText) > 0:
            newText += " " + correct(subword)
        else:
            newText = correct(subword)
    return newText


# สร้างออบเจ็กต์ PyAudio
audio = pyaudio.PyAudio()

# เปิดอุปกรณ์เสียง (ไมโครโฟน)
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

def speechToText_th():   
     
    # print("กำลังบันทึกเสียง...")
    frames = []
    silence_frames = 0
    
    # บันทึกเสียงจนกว่าจะหยุดพูดหรือหยุดเงียบเกิน 1 วินาที
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        rms = audioop.rms(data, 2)
        
        # ถ้าระดับเสียงต่ำกว่าค่าทศนิยมคงที่เป็นเวลา 1 วินาที ให้หยุดการบันทึก
        if rms < SILENT_THRESHOLD:
            silence_frames += 1
        else:
            silence_frames = 0
        
        if silence_frames / (RATE / CHUNK) > MAX_SILENCE_SECONDS:
            break

    # print("หยุดการบันทึกเสียง")
    # Convert bytes to tensor and load audio
    audio_data = b''.join(frames)

    wav_data = write_header(audio_data, 1, 2, 16_000)
    raw_audio, _ = sf.read(io.BytesIO(wav_data))
    # Preprocess the input for inference
    inputs = processor(raw_audio, sampling_rate=16_000, return_tensors="pt", padding=True)
    
    # แปลงน้ำหนักของโมเดลให้อยู่ในรูปแบบของ torch.cuda.FloatTensor
    model.to(dev)

    with torch.no_grad():
        # ทำการประมวลผลบน GPU
        logits = model(inputs.input_values.to(dev)).logits.cpu()


    predicted_ids = torch.argmax(logits, dim=-1)

    # Decode the predicted ids to text
    transcriptions = processor.batch_decode(predicted_ids)
    new_word = word_correction(transcriptions[0])
    return new_word


while True:
    test = speechToText_th()
    print(test)