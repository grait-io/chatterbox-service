//! Streaming TTS Example
//!
//! This example demonstrates how to use audio chunking for streaming
//! applications where you want to start playback before the full
//! audio is generated.
//!
//! # Running
//!
//! ```bash
//! cargo run --example streaming --features server
//! ```

use chatterbox_tts::{audio::AudioBuffer, AudioFormat, ChatterboxTts, Result, VoiceConfig};

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("=== Chatterbox Streaming TTS Example ===\n");

    // Create TTS instance
    let mut tts = ChatterboxTts::new()?;
    tts.load_model()?;
    tts.warmup()?;

    // Generate a longer piece of text
    let text = "This is a demonstration of streaming text to speech. \
                When you have a longer piece of text to synthesize, \
                you may want to start playing the audio before the entire \
                generation is complete. This example shows how to chunk \
                the generated audio for streaming playback.";

    println!("Generating speech for streaming...");
    let audio = tts.synthesize(text)?;

    println!("Total audio duration: {:.2}s", audio.duration_seconds);
    println!("Total samples: {}\n", audio.samples.len());

    // Method 1: Chunk into 40ms pieces (common for real-time streaming)
    println!("Method 1: 40ms chunks (PCM S16LE)");
    println!("-----------------------------------");

    let chunks = audio.chunk_pcm(40);
    println!("Number of chunks: {}", chunks.len());
    println!("Chunk size: {} bytes each", chunks.first().map(|c| c.len()).unwrap_or(0));
    println!("Total bytes: {}\n", chunks.iter().map(|c| c.len()).sum::<usize>());

    // Method 2: Using AudioBuffer for progressive streaming
    println!("Method 2: AudioBuffer for progressive streaming");
    println!("-------------------------------------------------");

    let mut buffer = AudioBuffer::new(AudioFormat::PcmS16Le, 40, audio.sample_rate);

    // Simulate receiving audio in chunks (e.g., from a model)
    let chunk_size = audio.samples.len() / 10;
    let mut total_chunks_sent = 0;

    for (i, samples) in audio.samples.chunks(chunk_size).enumerate() {
        // Add samples to buffer
        buffer.push_samples(samples);

        // Get available chunks to send
        let ready_chunks = buffer.drain_chunks();

        if !ready_chunks.is_empty() {
            println!(
                "  Batch {}: Received {} samples, sending {} chunks",
                i + 1,
                samples.len(),
                ready_chunks.len()
            );
            total_chunks_sent += ready_chunks.len();
        }
    }

    // Get final chunk
    if let Some(final_chunk) = buffer.finalize() {
        println!("  Final chunk: {} bytes (is_last={})", final_chunk.data.len(), final_chunk.is_last);
        total_chunks_sent += 1;
    }

    println!("\nTotal chunks sent: {}", total_chunks_sent);

    // Method 3: Different chunk sizes for different use cases
    println!("\nMethod 3: Different chunk sizes");
    println!("---------------------------------");

    let sizes = [20, 40, 100, 200];
    for ms in sizes {
        let chunks = audio.chunk(ms);
        let samples_per_chunk = chunks.first().map(|c| c.len()).unwrap_or(0);
        println!(
            "  {}ms chunks: {} chunks, {} samples each",
            ms,
            chunks.len(),
            samples_per_chunk
        );
    }

    // Save chunked audio to verify it reassembles correctly
    println!("\nSaving chunked audio to verify integrity...");
    let all_chunks = audio.chunk_pcm(40);
    let reassembled: Vec<u8> = all_chunks.into_iter().flatten().collect();

    // Write raw PCM
    std::fs::write("streaming_output.pcm", &reassembled)?;
    println!("Saved raw PCM to: streaming_output.pcm");

    // Also save as WAV for easy playback
    audio.save_wav("streaming_output.wav")?;
    println!("Saved WAV to: streaming_output.wav");

    println!("\n=== Done! ===");

    Ok(())
}
