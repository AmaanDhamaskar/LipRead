import moviepy.editor as mp
import webrtcvad
import numpy as np
import scipy.io.wavfile as wavfile
from pydub import AudioSegment
import io

def generate_alignments():

    # Usage example
    # Usage example
    video_path = 'input.mp4'
    alignment_path = 'input.align'

    # Load the video
    video = mp.VideoFileClip(video_path)
    video_duration = video.duration
    frame_rate = video.fps

    # Initialize the alignment file
    with open(alignment_path, 'w') as align_file:
        align_file.write('# Video Duration: {:.2f} seconds\n'.format(video_duration))

    # Initialize the WebRTC VAD
    vad = webrtcvad.Vad()

    # Set the aggressiveness level of VAD (0 to 3)
    vad.set_mode(3)

    # Process each frame in the video
    for t in range(int(video_duration * frame_rate)):
        # Extract the frame at time t
        frame = video.get_frame(t / frame_rate)

        # Convert frame to mono audio array
        audio_array = np.mean(frame, axis=1).astype(np.int16)

        # Save the mono audio array as a temporary WAV file
        temp_path = 'temp.wav'
        wavfile.write(temp_path, int(frame_rate), audio_array)

        # Load the temporary WAV file as an AudioSegment
        audio = AudioSegment.from_file(temp_path, format='wav')

        # Create a seekable file-like object from audio data
        audio_file = io.BytesIO()
        audio.export(audio_file, format='wav')
        audio_file.seek(0)

        # Load audio segment and apply VAD
        is_speech = vad.is_speech(audio_file.read(), sample_rate=audio.frame_rate)

        # Determine the alignment label based on speech detection
        alignment = "SPEECH" if is_speech else "SILENCE"

        # Write the alignment to the file
        with open(alignment_path, 'a') as align_file:
            align_file.write('{:.2f}: {}\n'.format(t / frame_rate, alignment))

    print('Alignments generated successfully.')
    return 'input.align'

