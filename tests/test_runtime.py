from __future__ import annotations

from textwrap import dedent

from moviepy.video.io.ffmpeg_reader import FFmpegInfosParser

from tribev2.runtime import apply_warning_filters


def test_moviepy_ffmpeg_parser_handles_multiline_displaymatrix_metadata() -> None:
    apply_warning_filters()

    infos = dedent(
        """
        Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'source.mov':
          Metadata:
            major_brand     : qt
          Duration: 00:00:09.54, start: 0.000000, bitrate: 112667 kb/s
          Stream #0:0(und): Video: hevc (Main 10) (hvc1 / 0x31637668), yuv420p10le(tv, bt2020nc/bt2020/arib-std-b67), 3840x2160, 101901 kb/s, 57.16 fps, 60 tbr, 600 tbn (default)
              Metadata:
                creation_time   : 2026-03-13T09:09:24.000000Z
                handler_name    : Core Media Video
              Side data:
                displaymatrix: rotation of -90.00 degrees
                Ambient Viewing Environment, ambient_illuminance=314.000000, ambient_light_x=0.312700, ambient_light_y=0.329000
        """
    ).strip()

    parsed = FFmpegInfosParser(
        infos,
        "source.mov",
        fps_source="fps",
        check_duration=True,
        decode_file=False,
    ).parse()

    assert parsed["video_rotation"] == 90.0
    assert parsed["inputs"][0]["streams"][0]["metadata"]["displaymatrix"] == "90.0\n"
