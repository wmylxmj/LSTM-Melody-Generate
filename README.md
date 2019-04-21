# LSTM-Melody-Generate
LSTM 旋律生成 2019-4-16
### Midi采样
- import parsers
- parser = parsers.MidiParser()
- sequence = parser.parse("midi file path")
##### 可得到如下序列
![](./images/midi.png)
### 单音旋律提取
- import parsers
- parser = parsers.MidiParser()
- sequence = parser.parse("midi file path")
- parser = parsers.get_monosyllabic_melody(sequence)
##### 可得到如下单音旋律序列
![](./images/melody.png)
