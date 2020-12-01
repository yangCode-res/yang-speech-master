import pyaudio
import wave
import time
nowtime=time.strftime("%H_%M_%S")
input_filename = nowtime+".wav"               # 麦克风采集的语音输入
input_filepath = "D:\\yang-speech-master\\result\\"              # 输入文件的path
in_path = input_filepath + input_filename


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
        exit()
    else:
        print("无效输入，请重新选择")
        get_audio(in_path)


get_audio(in_path)


# import numpy as np
# from scipy.io import wavfile
# import matplotlib.pyplot as plt
#
# sampling_freq, audio = wavfile.read(r"D:\Python\software.wav")   # 读取文件
#
# audio = audio / np.max(audio)   # 归一化，标准化
#
# # 应用傅里叶变换
# fft_signal = np.fft.fft(audio)
# print(fft_signal)
# # [-0.04022912+0.j         -0.04068997-0.00052721j -0.03933007-0.00448355j
# #  ... -0.03947908+0.00298096j -0.03933007+0.00448355j -0.04068997+0.00052721j]
#
# fft_signal = abs(fft_signal)
# print(fft_signal)
# # [0.04022912 0.04069339 0.0395848  ... 0.08001755 0.09203427 0.12889393]
#
# # 建立时间轴
# Freq = np.arange(0, len(fft_signal))
#
# # 绘制语音信号的
# plt.figure()
# plt.plot(Freq, fft_signal, color='blue')
# plt.xlabel('Freq (in kHz)')
# plt.ylabel('Amplitude')
# plt.show()