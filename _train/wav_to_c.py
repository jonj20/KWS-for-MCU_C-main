import wave

def wav_to_c_array(input_wav_file, output_c_file):
    with wave.open(input_wav_file, 'rb') as wav_file:
        # 获取音频参数
        n_channels = wav_file.getnchannels()
        samp_width = wav_file.getsampwidth()
        n_frames = wav_file.getnframes()
        data = wav_file.readframes(n_frames)

    # 转换为 bytearray 并限制在 uint8 范围
    byte_data = bytearray(data)

    # 写入 C 数组文件
    with open(output_c_file, 'w') as f:
        f.write(f"const unsigned char sound_data[{len(byte_data)}] = {{\n")
        for i in range(0, len(byte_data), 12):
            chunk = byte_data[i:i+12]
            line = ', '.join(f'0x{b:02x}' for b in chunk)
            f.write('    ' + line + ',\n')
        f.write("};\n")


wav_to_c_array("test_yes.wav", "output_sound.c")