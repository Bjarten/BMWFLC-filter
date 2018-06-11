import os

def save():
    os.system("ffmpeg -r 25 -framerate 25 -i  Images/%*.png -vcodec mpeg4 -y movie.mp4")

save( )