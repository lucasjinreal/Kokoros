use crate::onn::ort_koko::{self, ModelStrategy};
use crate::tts::tokenize::tokenize;
use crate::utils;
use crate::utils::debug::format_debug_prefix;
use lazy_static::lazy_static;
use ndarray::Array3;
use ndarray_npy::NpzReader;
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use espeak_rs::text_to_phonemes;

// Global mutex to serialize espeak-rs calls to prevent phoneme randomization
// espeak-rs uses global state internally and is not thread-safe
lazy_static! {
    static ref ESPEAK_MUTEX: Mutex<()> = Mutex::new(());
}

// Flag to ensure voice styles are only logged once
static VOICES_LOGGED: AtomicBool = AtomicBool::new(false);

#[derive(Debug, Clone)]
pub struct WordAlignment {
    pub word: String,
    pub start_sec: f32,
    pub end_sec: f32,
}

#[derive(Debug, Clone)]
pub enum TtsOutput {
    /// Standard audio, no timing data
    Audio(Vec<f32>),
    /// Audio with synchronized word timestamps
    Aligned(Vec<f32>, Vec<WordAlignment>),
}

impl TtsOutput {
    pub fn raw_output(self) -> (Vec<f32>, Option<Vec<WordAlignment>>) {
        match self {
            TtsOutput::Audio(a) => (a, None),
            TtsOutput::Aligned(a, b) => (a, Some(b)),
        }
    }
}

enum ExecutionMode<'a> {
    /// Collects all data, adjusts timestamps to be global, returns it at the end.
    Batch,
    /// Yields chunks immediately with relative timestamps. Returns None at end.
    Stream(&'a mut dyn FnMut(TtsOutput) -> Result<(), Box<dyn std::error::Error>>),
}

#[derive(Debug, Clone)]
pub struct TTSOpts<'a> {
    pub txt: &'a str,
    pub lan: &'a str,
    pub style_name: &'a str,
    pub save_path: &'a str,
    pub mono: bool,
    pub speed: f32,
    pub initial_silence: Option<usize>,
}

#[derive(Clone)]
pub struct TTSKoko {
    #[allow(dead_code)]
    model_path: String,
    model: Arc<Mutex<ort_koko::OrtKoko>>,
    styles: HashMap<String, Vec<[[f32; 256]; 1]>>,
    init_config: InitConfig,
}

/// Parallel TTS with multiple ONNX instances for true concurrency
#[derive(Clone)]
pub struct TTSKokoParallel {
    #[allow(dead_code)]
    model_path: String,
    models: Vec<Arc<Mutex<ort_koko::OrtKoko>>>,
    styles: HashMap<String, Vec<[[f32; 256]; 1]>>,
    init_config: InitConfig,
}

#[derive(Clone)]
pub struct InitConfig {
    pub model_url: String,
    pub voices_url: String,
    pub sample_rate: u32,
}

impl Default for InitConfig {
    fn default() -> Self {
        Self {
            model_url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx".into(),
            voices_url: "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin".into(),
            sample_rate: 24000,
        }
    }
}

impl TTSKoko {
    pub async fn new(model_path: &str, voices_path: &str) -> Self {
        Self::from_config(model_path, voices_path, InitConfig::default()).await
    }

    pub async fn from_config(model_path: &str, voices_path: &str, cfg: InitConfig) -> Self {
        if !Path::new(model_path).exists() {
            utils::fileio::download_file_from_url(cfg.model_url.as_str(), model_path)
                .await
                .expect("download model failed.");
        }

        if !Path::new(voices_path).exists() {
            utils::fileio::download_file_from_url(cfg.voices_url.as_str(), voices_path)
                .await
                .expect("download voices data file failed.");
        }

        let model = Arc::new(Mutex::new(
            ort_koko::OrtKoko::new(model_path.to_string())
                .expect("Failed to create Kokoro TTS model"),
        ));
        // TODO: if(not streaming) { model.print_info(); }
        // model.print_info();

        let styles = Self::load_voices(voices_path);

        TTSKoko {
            model_path: model_path.to_string(),
            model,
            styles,
            init_config: cfg,
        }
    }

    fn process_internal(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number_start: Option<usize>,
        mut mode: ExecutionMode,
    ) -> Result<Option<(Vec<f32>, Vec<WordAlignment>)>, Box<dyn std::error::Error>> {
        let chunks = self.split_text_into_chunks(txt, 500, lan);
      
        let start_chunk_num = chunk_number_start.unwrap_or(0);

        let debug_prefix = format_debug_prefix(request_id, instance_id);

        let process_one_chunk = |chunk: &str,
                                 chunk_num: usize|
         -> Result<TtsOutput, Box<dyn std::error::Error>> {
            let chunk_info = format!("Chunk: {}, ", chunk_num);
            tracing::debug!("{} {}text: '{}'", debug_prefix, chunk_info, chunk);

            // A. Tokenize
            // Only build the expensive alignment map if the loaded model supports timestamps.
            let use_alignment = {
                let model = self.model.lock().unwrap();
                matches!(model.strategy(), Some(ModelStrategy::Timestamped(_)))
            };

            let (mut tokens, word_map) = if use_alignment {
                self.tokenize_with_alignment(chunk, lan)
            } else {
                // Fast path for audio-only models: single eSpeak pass, no per-item calls
                self.tokenize_full_no_alignment(chunk, lan)
            };

            // Log token count (helpful for debugging context limits)
            tracing::debug!(
                "{} {}tokens generated: {}",
                debug_prefix,
                chunk_info,
                tokens.len()
            );

            // B. Silence
            let silence_count = initial_silence.unwrap_or(0);
            for _ in 0..silence_count {
                tokens.insert(0, 30);
            }

            // C. Style
            let styles = self.mix_styles(style_name, tokens.len())?;

            // D. Padding
            let mut padded_tokens = vec![0];
            padded_tokens.extend(tokens);
            padded_tokens.push(0);

            let index_offset = 1 + silence_count;
            let tokens_batch = vec![padded_tokens];

            // E. Infer
            let (chunk_audio_array, chunk_durations_opt) = self.model.lock().unwrap().infer(
                tokens_batch,
                styles,
                speed,
                request_id,
                instance_id,
                Some(chunk_num),
            )?;

            let chunk_audio: Vec<f32> = chunk_audio_array.iter().cloned().collect();

            // F. Calculate Alignments
            if let Some(durations) = chunk_durations_opt {
                let mut alignments = Vec::new();

                // Model durations are in frames (hop=600 @ 24 kHz) ⇒ 40 frames/sec.
                let frames_per_sec: f32 = 40.0;

                // Guard speed to avoid division by zero; timestamps should reflect the final render timeline.
                let speed_safe = if speed > 1e-6 { speed } else { 1.0 };

                // Include initial "silence tokens" time into the local time cursor. You already shift the
                // durations index by `index_offset = 1 + silence_count`; here we also advance the cursor by
                // the skipped frames so the first word starts at the actual audio time.
                let mut chunk_time_cursor_frames: f32 = 0.0;
                if silence_count > 0 {
                    let start = 1; // skip BOS
                    let end = (1 + silence_count).min(durations.len());
                    if end > start {
                        let silence_frames: f32 = durations[start..end].iter().sum();
                        chunk_time_cursor_frames += silence_frames;
                    }
                }

                // Punctuation pause table in seconds (tune as needed). We scale by 1/speed so faster speech shortens pauses.
                let punct_pause_s = |label: &str| -> f32 {
                    match label {
                        "." | "!" | "?" => 0.300, // 300 ms
                        "," => 0.150,             // 150 ms
                        ";" | ":" => 0.200,
                        _ => 0.0,
                    }
                };

                for (word, start, end) in word_map {
                    let adj_start = start + index_offset;
                    let adj_end = end + index_offset;

                    // Punctuation items are separate in word_map with zero token span; account for pause.
                    let is_punct = word.len() == 1 && ".,!?:;!?".contains(word.as_str());
                    if is_punct {
                        // Scale pauses by 1/speed so timestamps match rendered audio when speech rate changes.
                        let pause_s = punct_pause_s(&word) / speed_safe;
                        let pause_frames = pause_s * frames_per_sec;
                        let start_sec = chunk_time_cursor_frames / frames_per_sec;
                        let end_sec = (chunk_time_cursor_frames + pause_frames) / frames_per_sec;
                        alignments.push(WordAlignment {
                            word: word.clone(),
                            start_sec,
                            end_sec,
                        });
                        chunk_time_cursor_frames += pause_frames;
                        continue;
                    }

                    // Normal word span: sum its frame durations and advance the cursor.
                    if adj_start < adj_end && adj_end <= durations.len() {
                        let mut word_frames: f32 = durations[adj_start..adj_end].iter().sum();

                        // If your ONNX `durations` do NOT already include speed scaling, uncomment this line:
                        // word_frames /= speed_safe;
                        // (Leave it commented if the model already produces speed‑scaled durations.)

                        let start_sec = chunk_time_cursor_frames / frames_per_sec;
                        let end_sec = (chunk_time_cursor_frames + word_frames) / frames_per_sec;
                        alignments.push(WordAlignment {
                            word,
                            start_sec,
                            end_sec,
                        });
                        chunk_time_cursor_frames += word_frames;
                    }
                }

                // Per‑chunk closure: linearly scale the local alignment times to match this chunk’s audio length.
                // This eliminates cumulative drift across chunks and prevents middle events from sliding late.
                let t_end_sec = chunk_time_cursor_frames / frames_per_sec; // alignment‑derived duration (sec)
                let chunk_audio_sec = chunk_audio.len() as f32 / 24_000.0; // audio duration (sec)

                if t_end_sec > 0.0 {
                    let s = (chunk_audio_sec / t_end_sec);
                    // Optionally clamp extreme corrections; typical values should be close to 1.0
                    let s_clamped = s.clamp(0.8, 1.25);
                    if (s_clamped - 1.0).abs() > 0.005 {
                        // >0.5% correction
                        tracing::debug!(
                            scale = s_clamped,
                            "Per-chunk alignment scaling applied (speed-aware)"
                        );
                        for al in &mut alignments {
                            al.start_sec *= s_clamped;
                            al.end_sec *= s_clamped;
                        }
                    }

                    // Optional sanity log after scaling
                    let diff_ms = (((t_end_sec * s_clamped) - chunk_audio_sec) * 1000.0).abs();
                    if diff_ms > 10.0 {
                        tracing::warn!(
                            chunk_t_end_sec = t_end_sec * s_clamped,
                            chunk_audio_sec,
                            diff_ms,
                            "Alignment vs audio duration still off after scaling",
                        );
                    } else {
                        tracing::debug!(
                            chunk_t_end_sec = t_end_sec * s_clamped,
                            chunk_audio_sec,
                            "Chunk alignment closure OK",
                        );
                    }
                }

                Ok(TtsOutput::Aligned(chunk_audio, alignments))
            } else {
                Ok(TtsOutput::Audio(chunk_audio))
            }
        };

        match &mut mode {
            ExecutionMode::Stream(callback) => {
                for (i, chunk) in chunks.iter().enumerate() {
                    let output = process_one_chunk(chunk, start_chunk_num + i)?;
                    callback(output)?;
                }
                Ok(None)
            }

            ExecutionMode::Batch => {
                let mut batch_audio = Vec::new();
                let mut batch_alignments = Vec::new();
                let mut global_time_offset = 0.0;
                let sample_rate = 24000.0;

                for (i, chunk) in chunks.iter().enumerate() {
                    let output = process_one_chunk(chunk, start_chunk_num + i)?;

                    match output {
                        TtsOutput::Aligned(audio, alignments) => {
                            let duration = audio.len() as f32 / sample_rate;
                            batch_audio.extend_from_slice(&audio);

                            for mut align in alignments {
                                align.start_sec += global_time_offset;
                                align.end_sec += global_time_offset;
                                batch_alignments.push(align);
                            }
                            global_time_offset += duration;
                        }
                        TtsOutput::Audio(audio) => {
                            let duration = audio.len() as f32 / sample_rate;
                            batch_audio.extend_from_slice(&audio);
                            global_time_offset += duration;
                        }
                    }
                }
                Ok(Some((batch_audio, batch_alignments)))
            }
        }
    }

    /// Prosody-Aware Tokenization ---
    fn tokenize_with_alignment(
        &self,
        text: &str,
        lan: &str,
    ) -> (Vec<i64>, Vec<(String, usize, usize)>) {
        // We will produce tokens from the full, context-aware phonemes (best prosody)
        // and build an alignment map by estimating per-word token spans using
        // per-word phoneme tokenization. This keeps audio natural while providing
        // robust timestamps even when eSpeak merges words (e.g., "the model").

        // 1) Full-phrase phonemes and tokens (prosody source)
        let full_phonemes = {
            let _guard = ESPEAK_MUTEX.lock().unwrap();
            text_to_phonemes(text, lan, None, true, false)
                .unwrap_or_default()
                .join("")
        };
        let all_tokens = tokenize(&full_phonemes);

        // 2) Build a tokenization plan per original "word or punctuation" unit.
        //    We want punctuation timestamps too, so we split words and punctuation as separate items.
        //    Simple heuristic: split on whitespace, then further split trailing/leading punctuation
        //    for .,!?;: characters.
        fn split_words_and_punct(s: &str) -> Vec<String> {
            let mut out = Vec::new();
            for raw in s.split_whitespace() {
                let chars: Vec<char> = raw.chars().collect();
                let mut start = 0usize;
                let mut end = chars.len();

                // Leading punctuation
                while start < end {
                    let c = chars[start];
                    if ".,!?:;".contains(c) {
                        out.push(c.to_string());
                        start += 1;
                    } else {
                        break;
                    }
                }
                // Trailing punctuation
                while end > start {
                    let c = chars[end - 1];
                    if ".,!?:;".contains(c) {
                        end -= 1;
                    } else {
                        break;
                    }
                }
                if start < end {
                    out.push(chars[start..end].iter().collect());
                }
                // Push trailing punctuation in original order
                for i in end..chars.len() {
                    out.push(chars[i].to_string());
                }
            }
            out
        }

        let items = split_words_and_punct(text);

        // 3) For each item, get its standalone phonemes and token count.
        //    Punctuation-only items get zero tokens but we still record them for timestamps.
        let mut per_item_token_counts: Vec<usize> = Vec::with_capacity(items.len());
        let mut per_item_is_punct: Vec<bool> = Vec::with_capacity(items.len());
        for it in &items {
            if it.len() == 1 && ".,!?:;".contains(it.chars().next().unwrap()) {
                per_item_token_counts.push(0);
                per_item_is_punct.push(true);
            } else {
                let ph = {
                    let _guard = ESPEAK_MUTEX.lock().unwrap();
                    text_to_phonemes(it, lan, None, true, false)
                        .unwrap_or_default()
                        .join("")
                };
                let cnt = tokenize(&ph).len();
                per_item_token_counts.push(cnt);
                per_item_is_punct.push(false);
            }
        }

        // 4) Map per-item token counts onto the full token sequence length.
        //    If sums differ (likely due to coarticulation/context differences),
        //    rescale the counts to match the full length, keeping the distribution similar.
        let target_len = all_tokens.len();
        let mut sum_counts: usize = per_item_token_counts.iter().sum();

        let mut adjusted_counts: Vec<usize> = per_item_token_counts.clone();
        if sum_counts != target_len && sum_counts > 0 {
            let scale = (target_len as f64) / (sum_counts as f64);
            let mut fractional: Vec<(usize, f64)> = Vec::with_capacity(adjusted_counts.len());
            let mut new_sum = 0usize;
            for (i, &c) in per_item_token_counts.iter().enumerate() {
                let scaled = (c as f64) * scale;
                let floored = scaled.floor() as usize;
                adjusted_counts[i] = floored;
                new_sum += floored;
                fractional.push((i, scaled - floored as f64));
            }
            // Distribute the remaining tokens to the largest fractional parts
            let mut remaining = target_len.saturating_sub(new_sum);
            fractional.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (i, _) in fractional {
                if remaining == 0 {
                    break;
                }
                adjusted_counts[i] += 1;
                remaining -= 1;
            }
            tracing::debug!(
                "Alignment: rescaled per-item token counts from {} to {} to match durations length {}.",
                sum_counts,
                adjusted_counts.iter().sum::<usize>(),
                target_len
            );
            sum_counts = adjusted_counts.iter().sum();
        }

        // 5) Build the word_map by assigning contiguous spans across the token stream.
        //    Punctuation items receive zero-length spans by design (timestamp markers).
        let mut word_map: Vec<(String, usize, usize)> = Vec::with_capacity(items.len());
        let mut cursor = 0usize;
        for (idx, item) in items.iter().enumerate() {
            let cnt = adjusted_counts.get(idx).copied().unwrap_or(0);
            if per_item_is_punct[idx] {
                // Zero-length marker at current cursor
                word_map.push((item.clone(), cursor, cursor));
            } else {
                let start_idx = cursor;
                let end_idx = cursor.saturating_add(cnt);
                word_map.push((item.clone(), start_idx, end_idx));
                cursor = end_idx;
            }
        }

        // If our mapping under-ran due to rounding issues, extend the last non-punct item to cover all tokens
        if cursor < target_len {
            if let Some(last_non_punct_pos) =
                (0..word_map.len()).rev().find(|&i| !(per_item_is_punct[i]))
            {
                let (w, s, _e) = &word_map[last_non_punct_pos];
                word_map[last_non_punct_pos] = (w.clone(), *s, target_len);
            }
        }

        // If there are absolutely no tokens (empty text), return empty mapping
        (all_tokens, word_map)
    }

    /// Fast tokenization path for audio-only models (no timestamps)
    /// Performs a single eSpeak phonemization for the full text and returns tokens with an empty word map.
    fn tokenize_full_no_alignment(
        &self,
        text: &str,
        lan: &str,
    ) -> (Vec<i64>, Vec<(String, usize, usize)>) {
        let full_phonemes = {
            let _guard = ESPEAK_MUTEX.lock().unwrap();
            text_to_phonemes(text, lan, None, true, false)
                .unwrap_or_default()
                .join("")
        };
        let all_tokens = tokenize(&full_phonemes);
        (all_tokens, Vec::new())
    }

    fn split_text_into_chunks(&self, text: &str, max_tokens: usize, lan: &str) -> Vec<String> {
        let mut chunks = Vec::new();

        // First split by sentences - using common sentence ending punctuation
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '?' || c == '!' || c == ';')
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut current_chunk = String::new();

        for sentence in sentences {
            // Clean up the sentence and add back punctuation
            let sentence = format!("{}.", sentence.trim());

            // Convert to phonemes to check token count
            let sentence_phonemes = {
                let _guard = ESPEAK_MUTEX.lock().unwrap();
                text_to_phonemes(&sentence, lan, None, true, false)
                    .unwrap_or_default()
                    .join("")
            };
            let token_count = tokenize(&sentence_phonemes).len();

            if token_count > max_tokens {
                // If single sentence is too long, split by words
                let words: Vec<&str> = sentence.split_whitespace().collect();
                let mut word_chunk = String::new();

                for word in words {
                    let test_chunk = if word_chunk.is_empty() {
                        word.to_string()
                    } else {
                        format!("{} {}", word_chunk, word)
                    };

                    let test_phonemes = {
                        let _guard = ESPEAK_MUTEX.lock().unwrap();
                        text_to_phonemes(&test_chunk, lan, None, true, false)
                            .unwrap_or_default()
                            .join("")
                    };
                    let test_tokens = tokenize(&test_phonemes).len();

                    if test_tokens > max_tokens {
                        if !word_chunk.is_empty() {
                            chunks.push(word_chunk);
                        }
                        word_chunk = word.to_string();
                    } else {
                        word_chunk = test_chunk;
                    }
                }

                if !word_chunk.is_empty() {
                    chunks.push(word_chunk);
                }
            } else if !current_chunk.is_empty() {
                // Try to append to current chunk
                let test_text = format!("{} {}", current_chunk, sentence);
                let test_phonemes = {
                    let _guard = ESPEAK_MUTEX.lock().unwrap();
                    text_to_phonemes(&test_text, lan, None, true, false)
                        .unwrap_or_default()
                        .join("")
                };
                let test_tokens = tokenize(&test_phonemes).len();

                if test_tokens > max_tokens {
                    // If combining would exceed limit, start new chunk
                    chunks.push(current_chunk);
                    current_chunk = sentence;
                } else {
                    current_chunk = test_text;
                }
            } else {
                current_chunk = sentence;
            }
        }

        // Add the last chunk if not empty
        if !current_chunk.is_empty() {
            chunks.push(current_chunk);
        }

        chunks
    }

    /// Smart word-based chunking for async streaming
    /// Creates chunks based on natural speech boundaries using word count and punctuation
    pub fn split_text_into_speech_chunks(&self, text: &str, max_words: usize) -> Vec<String> {
        let mut chunks = Vec::new();

        // Split by sentence-ending punctuation first
        let sentences: Vec<&str> = text
            .split(|c| c == '.' || c == '!' || c == '?')
            .filter(|s| !s.trim().is_empty())
            .collect();

        for sentence in sentences {
            let sentence = sentence.trim();
            if sentence.is_empty() {
                continue;
            }

            // Count words in this sentence
            let words: Vec<&str> = sentence.split_whitespace().collect();
            let word_count = words.len();

            if word_count <= max_words {
                // Small sentence - add as complete chunk (preserve original punctuation)
                chunks.push(format!("{}.", sentence));
            } else {
                // Large sentence - split by punctuation marks while preserving them
                let mut sub_clauses = Vec::new();
                let mut current_pos = 0;

                for (i, ch) in sentence.char_indices() {
                    if ch == ',' || ch == ';' || ch == ':' {
                        if i > current_pos {
                            let clause_with_punct = format!("{}{}", &sentence[current_pos..i], ch);
                            sub_clauses.push(clause_with_punct);
                        }
                        current_pos = i + 1;
                    }
                }

                // Add remaining text
                if current_pos < sentence.len() {
                    sub_clauses.push(sentence[current_pos..].to_string());
                }

                let sub_clauses: Vec<&str> = sub_clauses
                    .iter()
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .collect();

                let mut current_chunk = String::new();
                let mut current_word_count = 0;

                for clause in sub_clauses {
                    let clause = clause.trim();
                    let clause_words: Vec<&str> = clause.split_whitespace().collect();
                    let clause_word_count = clause_words.len();

                    if current_word_count + clause_word_count <= max_words {
                        // Add clause to current chunk (preserve original punctuation)
                        if current_chunk.is_empty() {
                            current_chunk = clause.to_string();
                        } else {
                            current_chunk = format!("{} {}", current_chunk, clause);
                        }
                        current_word_count += clause_word_count;
                    } else {
                        // Start new chunk (preserve original punctuation)
                        if !current_chunk.is_empty() {
                            chunks.push(current_chunk);
                        }
                        current_chunk = clause.to_string();
                        current_word_count = clause_word_count;
                    }
                }

                // Add final chunk (preserve original punctuation)
                if !current_chunk.is_empty() {
                    chunks.push(current_chunk);
                }
            }
        }

        // If no sentences found, fall back to word-based chunking
        if chunks.is_empty() {
            let words: Vec<&str> = text.split_whitespace().collect();
            let mut current_chunk = String::new();
            let mut current_word_count = 0;

            for word in words {
                if current_word_count + 1 <= max_words {
                    if current_chunk.is_empty() {
                        current_chunk = word.to_string();
                    } else {
                        current_chunk = format!("{} {}", current_chunk, word);
                    }
                    current_word_count += 1;
                } else {
                    if !current_chunk.is_empty() {
                        chunks.push(current_chunk);
                    }
                    current_chunk = word.to_string();
                    current_word_count = 1;
                }
            }

            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
            }
        }

        chunks
    }

    pub fn tts_timestamped_raw_audio(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
    ) -> Result<Option<(Vec<f32>, Vec<WordAlignment>)>, Box<dyn Error>> {
        self.process_internal(
            txt,
            lan,
            style_name,
            speed,
            initial_silence,
            request_id,
            instance_id,
            chunk_number,
            ExecutionMode::Batch,
        )
    }

    pub fn tts_raw_audio(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let audio = self.process_internal(
            txt,
            lan,
            style_name,
            speed,
            initial_silence,
            request_id,
            instance_id,
            chunk_number,
            ExecutionMode::Batch,
        )?;

        Ok(audio.unwrap().0)
    }

    /// Streaming version that yields audio chunks as they're generated
    pub fn tts_raw_audio_streaming<F>(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
        mut chunk_callback: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(Vec<f32>) -> Result<(), Box<dyn std::error::Error>>,
    {
        let mut adapter = |output: TtsOutput| -> Result<(), Box<dyn std::error::Error>> {
            chunk_callback(output.raw_output().0)
        };

        self.process_internal(
            txt,
            lan,
            style_name,
            speed,
            initial_silence,
            request_id,
            instance_id,
            chunk_number,
            // Pass the ADAPTER, not the original callback
            ExecutionMode::Stream(&mut adapter),
        )?;

        Ok(())
    }

    /// Streaming version that strictly requires a timestamped model.
    /// Yields audio chunks + alignment data via the callback as they are generated.
    pub fn tts_timestamped_raw_audio_streaming<F>(
        &self,
        txt: &str,
        lan: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
        mut chunk_callback: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        // CHANGE: Callback accepts TtsOutput instead of just Vec<f32>
        F: FnMut((Vec<f32>, Vec<WordAlignment>)) -> Result<(), Box<dyn std::error::Error>>,
    {
        let mut adapter = |output: TtsOutput| -> Result<(), Box<dyn std::error::Error>> {
            let audio = output.raw_output();
            chunk_callback((audio.0, audio.1.unwrap()))
        };

        self.process_internal(
            txt,
            lan,
            style_name,
            speed,
            initial_silence,
            request_id,
            instance_id,
            chunk_number,
            ExecutionMode::Stream(&mut adapter),
        )?;

        Ok(())
    }

    pub fn tts(
        &self,
        TTSOpts {
            txt,
            lan,
            style_name,
            save_path,
            mono,
            speed,
            initial_silence,
        }: TTSOpts,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let audio = self.tts_raw_audio(
            &txt,
            lan,
            style_name,
            speed,
            initial_silence,
            None,
            None,
            None,
        )?;

        // Save to file
        if mono {
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: self.init_config.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create(save_path, spec)?;
            for &sample in &audio {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        } else {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: self.init_config.sample_rate,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };

            let mut writer = hound::WavWriter::create(save_path, spec)?;
            for &sample in &audio {
                writer.write_sample(sample)?;
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        eprintln!("Audio saved to {}", save_path);
        Ok(())
    }

    pub fn mix_styles(
        &self,
        style_name: &str,
        tokens_len: usize,
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
        if !style_name.contains("+") {
            if let Some(style) = self.styles.get(style_name) {
                let styles = vec![style[tokens_len][0].to_vec()];
                Ok(styles)
            } else {
                Err(format!("can not found from styles_map: {}", style_name).into())
            }
        } else {
            eprintln!("parsing style mix");
            let styles: Vec<&str> = style_name.split('+').collect();

            let mut style_names = Vec::new();
            let mut style_portions = Vec::new();

            for style in styles {
                if let Some((name, portion)) = style.split_once('.') {
                    if let Ok(portion) = portion.parse::<f32>() {
                        style_names.push(name);
                        style_portions.push(portion * 0.1);
                    }
                }
            }
            eprintln!("styles: {:?}, portions: {:?}", style_names, style_portions);

            let mut blended_style = vec![vec![0.0; 256]; 1];

            for (name, portion) in style_names.iter().zip(style_portions.iter()) {
                if let Some(style) = self.styles.get(*name) {
                    let style_slice = &style[tokens_len][0]; // This is a [256] array
                    // Blend into the blended_style
                    for j in 0..256 {
                        blended_style[0][j] += style_slice[j] * portion;
                    }
                }
            }
            eprintln!("blended_style: {:?}", blended_style);
            Ok(blended_style)
        }
    }

    fn load_voices(voices_path: &str) -> HashMap<String, Vec<[[f32; 256]; 1]>> {
        let mut npz = NpzReader::new(File::open(voices_path).unwrap()).unwrap();
        let mut map = HashMap::new();

        for voice in npz.names().unwrap() {
            let voice_data: Result<Array3<f32>, _> = npz.by_name(&voice);
            let voice_data = voice_data.unwrap();
            let mut tensor = vec![[[0.0; 256]; 1]; 511];
            for (i, inner_value) in voice_data.outer_iter().enumerate() {
                for (j, inner_inner_value) in inner_value.outer_iter().enumerate() {
                    for (k, number) in inner_inner_value.iter().enumerate() {
                        tensor[i][j][k] = *number;
                    }
                }
            }
            map.insert(voice, tensor);
        }

        let _sorted_voices = {
            let mut voices = map.keys().collect::<Vec<_>>();
            voices.sort();

            // Only log voices once across all TTS instances
            if !VOICES_LOGGED.swap(true, Ordering::Relaxed) {
                tracing::info!("==========================================");
                tracing::info!("Voice styles loaded ({} total):", voices.len());
                tracing::info!("==========================================");

                // Group voices by prefix
                let mut grouped_voices: std::collections::BTreeMap<&str, Vec<&str>> =
                    std::collections::BTreeMap::new();
                for voice in &voices {
                    if let Some(prefix) = voice.get(0..2) {
                        grouped_voices
                            .entry(prefix)
                            .or_insert_with(Vec::new)
                            .push(voice);
                    }
                }

                for (prefix, voices_in_group) in grouped_voices {
                    let category = match prefix {
                        "af" => "American Female(af)",
                        "am" => "American Male(am)",
                        "bf" => "British Female(bf)",
                        "bm" => "British Male(bm)",
                        "ef" => "European Female(ef)",
                        "em" => "European Male(em)",
                        "ff" => "French Female(ff)",
                        "hf" => "Hindi Female(hf)",
                        "hm" => "Hindi Male(hm)",
                        "if" => "Italian Female(if)",
                        "im" => "Italian Male(im)",
                        "jf" => "Japanese Female(jf)",
                        "jm" => "Japanese Male(jm)",
                        "pf" => "Portuguese Female(pf)",
                        "pm" => "Portuguese Male(pm)",
                        "zf" => "Chinese Female(zf)",
                        "zm" => "Chinese Male(zm)",
                        _ => prefix,
                    };

                    let voices_str = voices_in_group.join(", ");
                    // Gray out the voice information
                    tracing::info!("\x1b[90m{}: {}\x1b[0m", category, voices_str);
                }

                tracing::info!("==========================================");
            }

            voices
        };

        map
    }

    // Returns a sorted list of available voice names
    pub fn get_available_voices(&self) -> Vec<String> {
        let mut voices: Vec<String> = self.styles.keys().cloned().collect();
        voices.sort();
        voices
    }
}

impl TTSKokoParallel {
    pub async fn new_with_instances(
        model_path: &str,
        voices_path: &str,
        num_instances: usize,
    ) -> Self {
        Self::from_config_with_instances(
            model_path,
            voices_path,
            InitConfig::default(),
            num_instances,
        )
        .await
    }

    pub async fn from_config_with_instances(
        model_path: &str,
        voices_path: &str,
        cfg: InitConfig,
        num_instances: usize,
    ) -> Self {
        if !Path::new(model_path).exists() {
            utils::fileio::download_file_from_url(cfg.model_url.as_str(), model_path)
                .await
                .expect("download model failed.");
        }

        if !Path::new(voices_path).exists() {
            utils::fileio::download_file_from_url(cfg.voices_url.as_str(), voices_path)
                .await
                .expect("download voices data file failed.");
        }

        // Create multiple ONNX model instances
        let mut models = Vec::new();
        for i in 0..num_instances {
            tracing::info!(
                "Creating TTS instance [{}] ({}/{})",
                format!("{:02x}", i),
                i + 1,
                num_instances
            );
            let model = Arc::new(Mutex::new(
                ort_koko::OrtKoko::new(model_path.to_string())
                    .expect("Failed to create Kokoro TTS model"),
            ));
            models.push(model);
        }

        let styles = TTSKoko::load_voices(voices_path);

        TTSKokoParallel {
            model_path: model_path.to_string(),
            models,
            styles,
            init_config: cfg,
        }
    }

    /// Get a specific model instance for a worker
    pub fn get_model_instance(&self, worker_id: usize) -> Arc<Mutex<ort_koko::OrtKoko>> {
        let index = worker_id % self.models.len();
        Arc::clone(&self.models[index])
    }

    /// HELPER: Create a lightweight wrapper for a specific model ---
    fn get_tts_wrapper(&self, model_instance: Arc<Mutex<ort_koko::OrtKoko>>) -> TTSKoko {
        TTSKoko {
            model_path: self.model_path.clone(),
            model: model_instance,
            // TODO: This clones the HashMap. In a future PR, wrap styles in Arc<>!
            styles: self.styles.clone(),
            init_config: self.init_config.clone(),
        }
    }

    /// TTS with timestamps for model instance
    pub fn tts_timestamped_raw_audio_with_instance(
        &self,
        text: &str,
        language: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
        model_instance: Arc<Mutex<ort_koko::OrtKoko>>,
    ) -> Result<Option<(Vec<f32>, Vec<WordAlignment>)>, Box<dyn Error>> {
        let wrapper = self.get_tts_wrapper(model_instance);
        wrapper.tts_timestamped_raw_audio(
            text,
            language,
            style_name,
            speed,
            initial_silence,
            request_id,
            instance_id,
            chunk_number,
        )
    }

    /// TTS processing with specific model instance (no global lock)
    pub fn tts_raw_audio_with_instance(
        &self,
        text: &str,
        language: &str,
        style_name: &str,
        speed: f32,
        initial_silence: Option<usize>,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
        model_instance: Arc<Mutex<ort_koko::OrtKoko>>,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let wrapper = self.get_tts_wrapper(model_instance);

        wrapper.tts_raw_audio(
            text,
            language,
            style_name,
            speed,
            initial_silence,
            request_id,
            instance_id,
            chunk_number,
        )
    }

    /// Forward compatibility - split text method
    pub fn split_text_into_speech_chunks(&self, text: &str, max_words: usize) -> Vec<String> {
        // Use TTSKoko's implementation for now - create temporary instance
        let temp_tts = TTSKoko {
            model_path: self.model_path.clone(),
            model: Arc::clone(&self.models[0]), // Just for interface compatibility
            styles: self.styles.clone(),
            init_config: self.init_config.clone(),
        };
        temp_tts.split_text_into_speech_chunks(text, max_words)
    }

    /// Get available voices
    pub fn get_available_voices(&self) -> Vec<String> {
        let mut voices: Vec<String> = self.styles.keys().cloned().collect();
        voices.sort();
        voices
    }
}
