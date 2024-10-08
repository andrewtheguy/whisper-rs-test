use hound::{self};
use reqwest::get;
use sqlx::{sqlite::{SqliteConnectOptions, SqliteQueryResult}, Pool, SqlitePool};
use tempfile::NamedTempFile;
use url::Url;

use crate::{config::Config, silero::{self, Silero}, streaming::streaming_url, utils};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState};

use std::{fs::{self}, io::Write, path::Path, sync::{Arc, Mutex}, time::{SystemTime, UNIX_EPOCH}};
use serde_json::json;

use zhconv::{zhconv, Variant};

enum State {
    NoSpeech,
    HasSpeech,
}

impl State {
    fn convert(has_speech: bool) -> State {
        match has_speech {
            true => State::HasSpeech,
            _ => State::NoSpeech,
        }
    }
}

const TARGET_SAMPLE_RATE: i64 = 16000;
const SAMPLE_SIZE: usize = 1024;

   
/*
The VAD predicts speech in a chunk of Linear Pulse Code Modulation (LPCM) encoded audio samples. These may be 8 or 16 bit integers or 32 bit floats.

The model is trained using chunk sizes of 256, 512, and 768 samples for an 8000 hz sample rate. It is trained using chunk sizes of 512, 768, 1024 samples for a 16,000 hz sample rate.
*/
 
async fn process_buffer_with_vad<F>(silero: &mut Silero,url: &str, mut f: F) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&Vec<i16>),
{
    //let target_sample_rate: i32 = 16000;


    let mut buf:Vec<i16> = Vec::new();
    //let mut num = 1;

    let min_speech_duration_seconds = 3.0;

    let mut has_speech = false;

    let mut prev_sample:Option<Vec<i16>> = None;

    //let mut has_speech_time = 0.0;

    let mut prev_state = State::NoSpeech;

    //let whisper_wrapper_ref = RefCell::new(whisper_wrapper);
    //let whisper_wrapper_ref2 = &whisper_wrapper;
    let closure_annotated = |samples: Vec<i16>| {
        eprintln!("Received sample size: {}", samples.len());
        //assert!(samples.len() as i32 == target_sample_rate); //make sure it is one second
        //let sample2 = samples.clone();
        //silero.reset();
        //let mut rng = rand::thread_rng();
        //let probability: f64 = rng.gen();
        let probability = silero.calc_level(&samples).unwrap();
        //let len_after_samples: i32 = (buf.len() + samples.len()).try_into().unwrap();
        eprintln!("buf.len() {}", buf.len());
        let seconds = buf.len() as f32 / TARGET_SAMPLE_RATE as f32;
        //eprintln!("len_after_samples / target_sample_rate {}",seconds);

        if probability > 0.5 {
            eprintln!("Chunk is speech: {}", probability);
            has_speech = true;
        } else {
            has_speech = false;
        }

        match prev_state {
            State::NoSpeech => {
                if has_speech {
                    eprintln!("Transitioning from no speech to speech");
                    // add previous sample if it exists
                    if let Some(prev_sample2) = &prev_sample {
                        buf.extend(prev_sample2);
                    }
                    // start to extend the buffer
                    buf.extend(&samples);
                } else {
                    eprintln!("Still No Speech");
                }
            },
            State::HasSpeech => {
                if seconds < min_speech_duration_seconds {
                    eprintln!("override to Continue to has speech because seconds < min_seconds {}", seconds);
                    has_speech = true;
                }
                if has_speech {
                    eprintln!("Continue to has speech");
                    // continue to extend the buffer
                    buf.extend(&samples);
                } else {
                    eprintln!("Transitioning from speech to no speech");
                    buf.extend(&samples);
                    //save the buffer if not empty
                    f(&buf);
                    buf.clear();
                    //num += 1;
                }
            }
        }

        prev_state = State::convert(has_speech);
        
        prev_sample = Some(samples);
    };

    streaming_url(url,TARGET_SAMPLE_RATE,SAMPLE_SIZE,closure_annotated).await?;

    if buf.len() > 0 {
        f(&buf);
        buf.clear();
        //num += 1;
    }


    Ok(())
}




fn sync_buf_to_file(buf: &Vec<i16>, file_name: &str) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(file_name, spec).unwrap();
    for sample in buf {
        writer.write_sample(*sample).unwrap();
    }
    writer.finalize().unwrap();
}


fn transcribe(state: &mut WhisperState, params: &whisper_rs::FullParams, samples: &Vec<i16>) {
    
    // Create a state
    let mut audio = vec![0.0f32; samples.len().try_into().unwrap()];

    whisper_rs::convert_integer_to_float_audio(&samples, &mut audio).expect("Conversion error");

    // Run the model.
    state.full(params.clone(), &audio[..]).expect("failed to run model");

    //eprintln!("{}",state.full_n_segments().expect("failed to get number of segments"));
    //samples.clear();
}

fn get_filename_from_url(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    
    // Parse the URL
    let parsed_url = Url::parse(url)?;
    let path = parsed_url.path();
    if let Some(filename) = Path::new(path).file_name(){
        Ok(filename.to_str().unwrap().to_string())
    }else{
        Err("Failed to get filename from URL".into())
    }
}

async fn download_to_temp_and_move(url: &str, destination: &str) -> Result<(), Box<dyn std::error::Error>> {

    // Create a temporary file. This file will be automatically deleted when dropped.
    let mut temp_file = NamedTempFile::new()?;

    // Download the file
    let mut response = get(url).await?;
    
    if response.status().is_success() {

        // Stream the response body and write it to the file chunk by chunk
        while let Some(chunk) = response.chunk().await? {
            temp_file.write_all(&chunk)?;
        }
        // Move the temp file to the destination only if the download was successful.
        temp_file.persist(destination)?; // Moves the file to the final destination
        eprintln!("File downloaded and moved to: {}", destination);
    } else {
        println!("Failed to download the file. Status: {}", response.status());
    }

    Ok(())
}

async fn get_silero() -> silero::Silero {

    let download_url = "https://github.com/snakers4/silero-vad/raw/refs/tags/v5.1/src/silero_vad/data/silero_vad.onnx";

    let model_local_directory = dirs::cache_dir().unwrap().join("whisper_transcribe_rs");
    fs::create_dir_all(&model_local_directory).unwrap();
    let file_name = get_filename_from_url(download_url).unwrap();
    let model_path = model_local_directory.join(file_name);
    if !model_path.exists() {
        eprintln!("Downloading model from {} to {}", download_url, model_path.to_str().unwrap());
        download_to_temp_and_move(download_url, model_path.to_str().unwrap()).await.unwrap();
    }

    let sample_rate = match TARGET_SAMPLE_RATE {
        8000 => utils::SampleRate::EightkHz,
        16000 => utils::SampleRate::SixteenkHz,
        _ => panic!("Unsupported sample rate. Expect 8 kHz or 16 kHz."),
    };

    let silero = silero::Silero::new(sample_rate, model_path).unwrap();
    silero
}

pub async fn stream_to_file(config: Config) -> Result<(), Box<dyn std::error::Error>>{
    //let url = "https://rthkradio2-live.akamaized.net/hls/live/2040078/radio2/master.m3u8";
    //let url = "https://www.am1430.net/wp-content/uploads/show/%E7%B9%BC%E7%BA%8C%E6%9C%89%E5%BF%83%E4%BA%BA/2023/2024-10-03.mp3";
    //println!("First argument: {}", first_argument);

    let url = config.url.as_str();

    let mut num = 1;
    let closure_annotated = |buf: &Vec<i16>| {
        let file_name = format!("tmp/predict.stream.speech.{}.wav", format!("{:0>3}",num));
        sync_buf_to_file(&buf, &file_name);
        num += 1;
    };

    let mut silero = get_silero().await;

    process_buffer_with_vad(&mut silero,url,closure_annotated).await?;

    Ok(())
}


pub async fn transcribe_url(config: Config) -> Result<(), Box<dyn std::error::Error>> {

    let url = config.url.as_str();

        // Create a connection pool to the SQLite database
        let pool = SqlitePool::connect_with(SqliteConnectOptions::new()
        .filename("./tmp/example.db")
        .create_if_missing(true)).await?;
        sqlx::query(
            "CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY,
                    timestamp datetime NOT NULL,
                    content TEXT NOT NULL
            )"
        ) .execute(&pool)
        .await?;

    


    // Load a context and model.
    let context_param = WhisperContextParameters::default();

    let download_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/4496f29dabb6f37d8e6c45c3ec89ccbe66a832ea/ggml-large-v3-turbo.bin?download=true";

    let model_local_directory = dirs::cache_dir().unwrap().join("whisper_transcribe_rs");
    fs::create_dir_all(&model_local_directory).unwrap();
    let file_name = get_filename_from_url(download_url).unwrap();
    let model_path = model_local_directory.join(file_name);
    if !model_path.exists() {
        eprintln!("Downloading model from {} to {}", download_url, model_path.to_str().unwrap());
        download_to_temp_and_move(download_url, model_path.to_str().unwrap()).await.unwrap();
    }

    let ctx = WhisperContext::new_with_params(
        model_path.to_str().unwrap(),
        context_param,
    ).expect("failed to load model");



    // Create a params object for running the model.
    // The number of past samples to consider defaults to 0.
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 5 });

    // Edit params as needed.
    // Set the number of threads to use to 4.
    params.set_n_threads(4);
    // Enable translation.
    params.set_translate(false);
    // Set the language to translate to to English.
    params.set_language(Some(config.language.as_str()));
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

    let rt = tokio::runtime::Handle::try_current()?;

    sqlx::query(
        r#"
        INSERT INTO transcripts (timestamp, content) VALUES ($1, $2)
        "#,
    )
    .bind(1)
    .bind("test")
    .execute(&pool).await.expect("failed to insert data into the table");

    let language = config.language.clone();
    let arc = Arc::new(pool);
    params.set_segment_callback_safe(   move |data: whisper_rs::SegmentCallbackData| {
        
        let start = SystemTime::now();
        let since_the_epoch = start
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards");

        let line = json!({"start_timestamp":data.start_timestamp,
            "end_timestamp":data.end_timestamp, "cur_ts": since_the_epoch.as_millis() as f64/1000.0, "text":data.text});
        println!("{}", line);

        // only convert to traditional chinese when saving to db
        // output original in jsonl
        let db_save_text = match language.as_str() {
            "zh" | "yue" => {
                zhconv(&data.text, Variant::ZhHant)
            },
            _ => {
                data.text
            }
        };
        
        let arc= arc.clone();

        rt.spawn(async {
            println!("now running on a worker thread");
        });

            let arc_clone = arc.clone();
            // Insert data into the table
            rt.spawn(async move {
                sqlx::query(
                r#"
                "INSERT INTO transcripts (timestamp, content) VALUES (?1, ?2)"
                "#,
            )
            .bind(since_the_epoch.as_millis() as f64/1000.0)
            .bind(db_save_text)
            .execute(arc_clone.as_ref()).await.expect("failed to insert data into the table");
        println!("inserted data into the table!!!");
            });
            //rt.block_on(fut).expect("failed to insert data into the table");
      
    

    });


    let mut state = ctx.create_state().expect("failed to create key");


    //let whisper_wrapper_ref = RefCell::new(whisper_wrapper);
    //let whisper_wrapper_ref2 = &whisper_wrapper;
    let closure_annotated = |buf: &Vec<i16>| {

            transcribe(&mut state, &params.clone(), &buf);

    };

    let mut silero = get_silero().await;

    process_buffer_with_vad(&mut silero,url,closure_annotated).await?;
        
    Ok(())
}