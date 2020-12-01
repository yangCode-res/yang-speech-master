#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from model_speech1 import ModelSpeech
from LanguageModel2 import ModelLanguage
from keras import backend as K
import pyaudio
import wave
import time

nowtime=time.strftime("%H_%M_%S")
input_filename = nowtime+".wav"               # 麦克风采集的语音输入
input_filepath = "D:\\yang-speech-master\\result\\"              # 输入文件的path
in_path = input_filepath + input_filename
datapath = ''
#modelpath = 'model_speech'
def get_audio(filepath):
    aa = str(input("是否开始录音？   （是/否）"))
    if aa == str("是") :
        CHUNK = 256
        FORMAT = pyaudio.paInt16
        CHANNELS = 1                # 声道数
        RATE = 16000                # 采样率
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = filepath
        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("*"*10, "开始录音：请在5秒内输入语音")
        frames = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("*"*10, "录音结束\n")
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    elif aa == str("否"):
        pass
    else:
        print("无效输入，请重新选择")
        get_audio(in_path)
def Recordingtotext():
    get_audio(in_path)
    datapath = 'D:\\ASRT_SpeechRecognition-master\\dataset'
    modelpath = 'model_speech'+ '\\'
    ms = ModelSpeech(datapath)
    ms.LoadModel('./m251/speech_model251_e_0_step_1361250.model')
    r = ms.RecognizeSpeech_FromFile('22_06_24.wav')
    K.clear_session()

    #print('*[提示] 语音识别结果：\n', r)

    ml = ModelLanguage('model_language')
    ml.LoadModel()

    str_pinyin = r
    r = ml.SpeechToText(str_pinyin)
    #print('语音转文字结果：\n', r)
    return r

if __name__ == '__main__':

    x =Recordingtotext()
    print(x)












