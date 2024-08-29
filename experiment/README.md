# visual-word-paradigm

## Setup

1. Clone this git repo
```
git clone git@github.com:cleresk/visual-world-paradigm.git
```

2. In the config.yml, enter a custom location for the experiments temp-Folder
```
directories:
temp: "C:\Users\Karl\MyTemp"
```
3. Start OpenSesame and open the experiment via the .osexp-File

4. Upload all the assets from the assets-folder to the OpenSesame file pool

5. Start the Eye-Tracker (recommended: 150 Hz)

5. Start the experiment

## Stimuli Requirements:

static:
- resolution: 300 x 300 px // 600 x 600 dpi
- color representation: sRGB
- format: JPG-Format (.jpg)

motion:
- resolution: 300 x 300 px
- format: MP4-Format (.mp4)
- framerate: minimum 25 FPS (frames per second)


## Requirements for audio:
- format: OGG-Format (.ogg)
- sampling rate: 48.000 kHz
- speed: 1x
- voice: tom
- generator: https://www.acoust.io/
