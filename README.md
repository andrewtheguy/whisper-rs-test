sample setup to transcribe with whisper.cpp's rust binding
it will convert the file with ffmpeg to the waveform in memory first
need to install ffmpeg separately

- create a config toml file
- then run
```
cargo run -- config.toml 2> >(rotatelogs -n 5 ./tmp/output.log 1M >&2)
```

windows:
```
cargo run -- config.toml 2>&1 | Where-Object { $_ -is [System.Management.Automation.ErrorRecord] } | ..\rotatelogs.exe -n 5 ./tmp/output.log 1M
```
- see config*.toml for config examples

- still need improvement on silero vad to include clips before and after speech/no speech transitions

- need to convert eprintln! to log!
