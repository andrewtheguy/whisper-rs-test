
use std::process::{Command, Stdio};
use std::io::Read;
use std::convert::TryInto;

pub mod streaming;
pub mod silero;
pub mod utils;
pub mod vad_processor;
pub mod config;

pub fn convert_file_to_wave(input_file: &str,target_sample_rate: i32) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    // Path to the input file
    //let input_file = "input.mp3"; // Replace with your file path

    // Run ffmpeg to get raw PCM (s16le) data at 16kHz
    let mut ffmpeg_process = Command::new("ffmpeg")
        .args(&[
            "-i", input_file,      // Input file
            "-f", "s16le",         // Output format: raw PCM, signed 16-bit little-endian
            "-acodec", "pcm_s16le",// Audio codec: PCM 16-bit signed little-endian
            "-ac", "1",            // Number of audio channels (1 = mono)
            "-ar", &format!("{}",target_sample_rate),        // Sample rate: 16 kHz
            "-"                    // Output to stdout
        ])
        .stdout(Stdio::piped())
        //.stderr(Stdio::null()) // Optional: Ignore stderr output
        .spawn()?;

    // Capture the stdout from the ffmpeg process (raw PCM data)
    let mut reader = std::io::BufReader::new(
        ffmpeg_process.stdout.take().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Failed to capture ffmpeg stdout"))?
    );
    let mut buffer: Vec<u8> = Vec::new();
    reader.read_to_end(&mut buffer)?;

    // Wait for the ffmpeg process to finish and check the exit status
    let status = ffmpeg_process.wait()?;
    if !status.success() {
        return Err(format!("ffmpeg failed with a non-zero exit code {}", status.code().unwrap_or(-1)).into());
    }

    // Convert the raw byte buffer into Vec<i16>
    let mut samples: Vec<i16> = Vec::with_capacity(buffer.len() / 2); // i16 is 2 bytes
    for chunk in buffer.chunks_exact(2) {
        let sample = i16::from_le_bytes(chunk.try_into().unwrap()); // Convert 2 bytes to i16
        samples.push(sample);
    }

    // `samples` now holds the audio data as `Vec<i16>` in 16 kHz
    println!("Captured {} samples at 16kHz", samples.len());

    Ok(samples)
}
