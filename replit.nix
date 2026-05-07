{ pkgs }: {
  deps = [
    pkgs.python312
    pkgs.python312Packages.pip
    pkgs.nodejs_20
    pkgs.gcc
    pkgs.ffmpeg
    pkgs.portaudio
    pkgs.libsndfile
    pkgs.pkg-config
  ];
}
