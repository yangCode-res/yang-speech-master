import wave
import numpy as np
import matplotlib.pyplot as plt
#import pylab as plt
from scipy.io import wavfile
import math
import time
from python_speech_features import mfcc




def import_data(filename):
    wav=wave.open(filename,"rb")# 打开一个wav文件
    num_frame=wav.getnframes()#获取wav文件的全部帧数
    num_channel=wav.getnchannels()#获取wave的声道数
    framerate=wav.getframerate()#获取帧速率
    str_data=wav.readframes(num_frame)#获取数据，，需要给定一个读取的长度，此时数据是字符串的数据
    wav.close()#关闭文件流
    wav_data=np.fromstring(str_data,dtype=np.short)#将声音文件数据转换为数组
    # 通过fromstring函数将字符串转换为数组，通过其参数dtype指定转换后的数据格式，由于我们的声音格式是以两个字节表示一个取
    # 样值，因此采用short数据类型转换。现在我们得到的wave_data是一个一维的short类型的数组，但是因为我们的声音文件是双声
    # 道的，因此它由左右两个声道的取样交替构成：LRLRLRLR....LR（L表示左声道的取样值，R表示右声道取样值）。修改wave_data
    # 的sharp之后：
    wav_data.shape=-1,num_channel#单声道时是一列数组，双声道的时候是两列的矩阵
    wav_data=wav_data.T#矩阵转置
    time=np.arange(0,num_frame)*(1.0/framerate)
    #通过取样点数和取样频率计算出每个取样的时间
    return wav_data[0],framerate,time #返回数组数据以及，帧速率
def wavetodatatest():
    dataed,fs,time=import_data("20170001P00001A0011.wav")
    plt.subplot(211)
    plt.plot(time,dataed[0])
    print(fs)
    #plt.subplot(212)
   # plt.plot(time,dataed[1],c="g")
    plt.xlabel("time(seconds)")
    plt.show()

# 绘制频域图
def plot_freq(signal, sample_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, sample_rate/2, fft_size/2 + 1)
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()
# 绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20, 5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()
# 绘制时域图
def plot_time(signal, sample_rate):

    time = np.arange(0, len(signal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, signal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
def get_wav_list(filename):
	'''
	读取一个wav文件列表，返回一个存储该列表的字典类型值
	ps:在数据中专门有几个文件用于存放用于训练、验证和测试的wav文件列表
	'''
	txt_obj=open(filename,'r') # 打开文件并读入
	txt_text=txt_obj.read()
	txt_lines=txt_text.split('\n') # 文本分割
	dic_filelist={} # 初始化字典
	list_wavmark=[] # 初始化wav列表
	for i in txt_lines:
		if(i!=''):
			txt_l=i.split(' ')
			dic_filelist[txt_l[0]] = txt_l[1]
			list_wavmark.append(txt_l[0])
	txt_obj.close()
	return dic_filelist,list_wavmark
def getFBankfeature(signal,sample_rate):
    signal = signal[0: int(3.5 * sample_rate)]  # Keep the first 3.5 seconds
    print('sample rate:', sample_rate, ', frame length:', len(signal))
    #预加重（Pre-Emphasis）
    #预加重一般是数字语音信号处理的第一步。语音信号往往会有频谱倾斜（Spectral Tilt）现象，
    # 即高频部分的幅度会比低频部分的小，
    # 预加重在这里就是起到一个平衡频谱的作用，增大高频部分的幅度。它使用如下的一阶滤波器来实现：
    pre_emphasis = 0.97
    emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    #分帧（Framing）
    #在预加重之后，需要将信号分成短时帧。做这一步的原因是：信号中的频率会随时间变化（不稳定的），
    # 一些信号处理算法（比如傅里叶变换）通常希望信号是稳定，也就是说对整个信号进行处理是没有意义的，
    # 因为信号的频率轮廓会随着时间的推移而丢失。为了避免这种情况，需要对信号进行分帧处理，
    # 认为每一帧之内的信号是短时不变的。一般设置帧长取20ms~40ms，相邻帧之间50%（+/-10%）的覆盖。
    # 对于ASR而言，通常取帧长为25ms，覆盖为10ms。

    frame_size, frame_stride = 0.025, 0.01
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    signal_length = len(emphasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1

    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.arange(0, frame_length).reshape(1, -1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,
                                                                                                                    1)
    frames = pad_signal[indices]
    #加窗（Window）
    #在分帧之后，通常需要对每帧的信号进行加窗处理。目的是让帧两端平滑地衰减，
    # 这样可以降低后续傅里叶变换后旁瓣的强度，取得更高质量的频谱。
    # 常用的窗有：矩形窗、汉明（Hamming）窗、汉宁窗（Hanning），以汉明窗为例
    hamming = np.hamming(frame_length)
    # hamming = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(0, frame_length) / (frame_length - 1))
    frames *= hamming
    #快速傅里叶变换（FFT）
    #对于每一帧的加窗信号，进行N点FFT变换，也称短时傅里叶变换（STFT），
    # N通常取256或512，
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    print(pow_frames.shape)
    #FBank特征
    #在介绍Mel滤波器组之前，先介绍一下Mel刻度，这是一个能模拟人耳接收声音规律的刻度，
    # 人耳在接收声音时呈现非线性状态，对高频的更不敏感，因此Mel刻度在低频区分辨度较高，
    # 在高频区分辨度较低，与频率之间的换算关系为：

    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    print(low_freq_mel, high_freq_mel)
    nfilt = 40
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # 所有的mel中心点，为了方便后面计算mel滤波器组，左右两边各补一个中心点
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))  # 各个mel滤波器在能量谱对应点的取值
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)  # 各个mel滤波器中心点对应FFT的区域编码，找到有值的位置
    for i in range(1, nfilt + 1):
        left = int(bin[i - 1])
        center = int(bin[i])
        right = int(bin[i + 1])
        for j in range(left, center):
            fbank[i - 1, j + 1] = (j + 1 - bin[i - 1]) / (bin[i] - bin[i - 1])
        for j in range(center, right):
            fbank[i - 1, j + 1] = (bin[i + 1] - (j + 1)) / (bin[i + 1] - bin[i])
    print(fbank)

    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # dB
    print(filter_banks.shape)
    plot_spectrogram(filter_banks.T, 'Filter Banks')

    pass
if  __name__ == '__main__':
    #sample_rate, signal = wavfile.read('D4_795.wav')
    signal,sample_rate,time=import_data('D4_795.wav')
    plot_time(signal,sample_rate)
    plot_freq(signal,sample_rate)
    getFBankfeature(signal,sample_rate)














