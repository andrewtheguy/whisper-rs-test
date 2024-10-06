use byteorder::{ByteOrder, LittleEndian};
use hound::{self, Sample};

use log4rs::append::file;
use serde_json::json;
use voice_activity_detector::{StreamExt as _, VoiceActivityDetector};
use tokio_stream::{self, StreamExt};
use tokio::io::{self, BufReader};
use tokio_util::{bytes::buf, io::ReaderStream};
use whisper_rs_test::streaming::streaming_url;
use tokio_util::{bytes::Bytes};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

#[derive(Debug, PartialEq, Eq)]
enum Operation {
    SplitFiles,
    Transcribe,
}

fn convert_to_i16_vec(buf: &[u8]) -> Vec<i16> {
    let mut vec = Vec::with_capacity(buf.len() / 2); // Allocate space for i16 values
    for chunk in buf.chunks_exact(2) {
        vec.push(LittleEndian::read_i16(chunk));
    }
    vec
}

fn transcribe(ctx: &WhisperContext, params: &whisper_rs::FullParams, samples: &Vec<i16>) {

    // Create a state
    let mut state = ctx.create_state().expect("failed to create key");

    let mut audio = vec![0.0f32; samples.len().try_into().unwrap()];

    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    // Run the model.
    state.full(params.clone(), &audio[..]).expect("failed to run model");

    //eprintln!("{}",state.full_n_segments().expect("failed to get number of segments"));
    //samples.clear();
}


/*
The VAD predicts speech in a chunk of Linear Pulse Code Modulation (LPCM) encoded audio samples. These may be 8 or 16 bit integers or 32 bit floats.

The model is trained using chunk sizes of 256, 512, and 768 samples for an 8000 hz sample rate. It is trained using chunk sizes of 512, 768, 1024 samples for a 16,000 hz sample rate.
*/
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target_sample_rate: i32 = 16000;

    let url = "https://rthkradio2-live.akamaized.net/hls/live/2040078/radio2/master.m3u8";
    //let url = "https://www.am1430.net/wp-content/uploads/show/%E7%B9%BC%E7%BA%8C%E6%9C%89%E5%BF%83%E4%BA%BA/2023/2024-10-03.mp3";
    //println!("First argument: {}", first_argument);


    let operation = Operation::SplitFiles;

    log4rs::init_file("log4rs.yaml", Default::default()).unwrap();

    whisper_rs::install_whisper_log_trampoline();

    // Load a context and model.
    let context_param = WhisperContextParameters::default();

    let ctx = WhisperContext::new_with_params(
        "/Users/it3/codes/andrew/transcribe_audio/whisper_models/ggml-large-v3-turbo.bin",
        context_param,
    )
        .expect("failed to load model");



    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });

    // Edit params as needed.
    // Set the number of threads to use to 4.
    params.set_n_threads(4);
    // Enable translation.
    params.set_translate(false);
    // Set the language to translate to to English.
    params.set_language(Some("yue"));
    // Disable anything that prints to stdout.
    params.set_print_special(false);
    params.set_debug_mode(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Enable token level timestamps
    params.set_token_timestamps(true);
    params.set_n_max_text_ctx(64);

    //let mut file = File::create("transcript.jsonl").expect("failed to create file");

    params.set_segment_callback_safe( move |data: whisper_rs::SegmentCallbackData| {
        let line = json!({"start_timestamp":data.start_timestamp,
            "end_timestamp":data.end_timestamp, "text":data.text});
        println!("{}", line);
        //writeln!(file, "{}", line).expect("failed to write to file");
    });


    //let samples = [0i16; 51200];
    let mut vad = VoiceActivityDetector::builder()
        .sample_rate(target_sample_rate)
        .chunk_size(512usize)
        .build()?;


    let mut buf:Vec<i16> = Vec::new();
    let mut num = 1;
    let max_seconds = 10;
    let mut last_uncommitted_no_speech: Option<Vec<i16>> = None;
    //let size_for_one_second = target_sample_rate * 2;
    //let cur_seconds = 0;

    //let whisper_wrapper_ref = RefCell::new(whisper_wrapper);
    //let whisper_wrapper_ref2 = &whisper_wrapper;
    let closure_annotated = |chunk: Vec<u8>| {

        eprintln!("Received chunk of size: {}", chunk.len());
        //assert!(chunk.len() as i32 == target_sample_rate * 2); //make sure it is one second
        //cur_seconds += 1;
        let samples = convert_to_i16_vec(&chunk);
        //assert!(samples.len() as i32 == target_sample_rate); //make sure it is one second
        let probability = vad.predict(samples.clone());
        let len_after_samples: i32 = (buf.len() + samples.len()).try_into().unwrap();
        if buf.len() > 0 && (len_after_samples / target_sample_rate) % max_seconds == 0 {
            eprintln!("Chunk is more than {} seconds, flushing", max_seconds);
            //add the last uncommitted no speech first
            if let Some(last_uncommitted_no_speech2) = &last_uncommitted_no_speech {
                buf.extend(last_uncommitted_no_speech2);
                last_uncommitted_no_speech = None;
            }
            buf.extend(&samples);
            transcribe(&ctx, &params.clone(), &mut buf);
            buf.clear();
            num += 1;
            //cur_seconds = 0;
        } else if probability > 0.5 {
            eprintln!("Chunk is speech: {}", probability);
            //add the last uncommitted no speech first
            if let Some(last_uncommitted_no_speech2) = &last_uncommitted_no_speech {
                buf.extend(last_uncommitted_no_speech2);
                last_uncommitted_no_speech = None;
            }
            buf.extend(&samples);
        } else {
            eprintln!("Chunk is not speech: {}", probability);
            if buf.len() > 0 {
                buf.extend(&samples);
                last_uncommitted_no_speech = None;
                transcribe(&ctx, &params.clone(), &mut buf);
                buf.clear();
                num += 1;
            }else{ //not committed yet
                last_uncommitted_no_speech = Some(samples);
            }
        }
    };

    streaming_url(url,target_sample_rate,Box::new(closure_annotated)).await?;

    if buf.len() > 0 {
        transcribe(&ctx, &params.clone(), &mut buf);
        buf.clear();
        num += 1;
    }


    Ok(())
}
