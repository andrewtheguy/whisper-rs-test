use std::env;
use std::fs;
use std::process;

use whisper_transcribe_rs::vad_processor::stream_to_file;
use whisper_transcribe_rs::vad_processor::transcribe_url;
use whisper_transcribe_rs::config::Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {


    // Collect command-line arguments, skipping the first one (program's name)
    let args: Vec<String> = env::args().collect();

    // Check if the first argument (after the program name) is provided
    if args.len() < 2 {
        eprintln!("Usage: {} config.toml", args[0]);
        process::exit(1);
    }

    // Get the first argument (the second element in the args vector)
    let config_file = &args[1];
    


    let config: Config = toml::from_str(fs::read_to_string(config_file)?.as_str()).unwrap();

    let operation = config.operation.as_str();

    match operation {
        "save_to_file"=>{
            //let url = "https://www.am1430.net/wp-content/uploads/show/%E7%B9%BC%E7%BA%8C%E6%9C%89%E5%BF%83%E4%BA%BA/2023/2024-10-03.mp3";
            stream_to_file(config).await?;
        },
        "transcribe"=>{
            //let url = "https://rthkradio2-live.akamaized.net/hls/live/2040078/radio2/master.m3u8";
            log4rs::init_file("log4rs.yaml", Default::default()).unwrap();
            whisper_rs::install_whisper_log_trampoline();
            transcribe_url(config).await?;
        },
        _=>{
            eprintln!("unknown operation: {}", operation);
            process::exit(1);
        }
    }

    
    Ok(())
}
