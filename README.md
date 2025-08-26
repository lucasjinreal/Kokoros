<div align="center">
  <img src="https://img2023.cnblogs.com/blog/3572323/202501/3572323-20250112184100378-907988670.jpg" alt="Banner" width="400" height="190">
</div>
<br>
<h1 align="center">🔥🔥🔥 Kokoro Rust</h1>

## [Zonos Rust Is On The Way?](https://github.com/lucasjinreal/Kokoros/issues/60)
## [Spark-TTS On The Way?](https://github.com/lucasjinreal/Kokoros/issues/75)
## [Orpheus-TTS On The Way?](https://github.com/lucasjinreal/Kokoros/issues/75)


**ASMR**

https://github.com/user-attachments/assets/1043dfd3-969f-4e10-8b56-daf8285e7420

(typo in video, ignore it)

**Digital Human**

https://github.com/user-attachments/assets/9f5e8fe9-d352-47a9-b4a1-418ec1769567

<p align="center">
  <b>Give a star ⭐ if you like it!</b>
</p>

[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) is a trending top 2 TTS model on huggingface.
This repo provides **insanely fast Kokoro infer in Rust**, you can now have your built TTS engine powered by Kokoro and infer fast by only a command of `koko`.

`kokoros` is a `rust` crate that provides easy to use TTS ability.
One can directly call `koko` in terminal to synthesize audio.

`kokoros` uses a relative small model 87M params, while results in extremly good quality voices results.

Languge support:

- [x] English;
- [x] Chinese (partly);
- [x] Japanese (partly);
- [x] German (partly);

> 🔥🔥🔥🔥🔥🔥🔥🔥🔥 Kokoros Rust version just got a lot attention now. If you also interested in insanely fast inference, embeded build, wasm support etc, please star this repo! We are keep updating it.

New Discord community: https://discord.gg/E566zfDWqD, Please join us if you interested in Rust Kokoro.

## Updates

- **_`2025.07.15`_**: 🔥🔥🔥 **CLI performance optimization and enhanced logging.** CLI now automatically uses optimal single-instance performance with intelligent CPU threading, comprehensive logging configuration options (`--log cli/file/all/none`), and simplified API with unified constructor patterns for better developer experience;
- **_`2025.07.12`_**: 🔥🔥🔥 **HTTP API streaming and parallel processing infrastructure.** OpenAI-compatible server supports streaming audio generation with `"stream": true` achieving 1-2s time-to-first-audio, work-in-progress parallel TTS processing with `--instances` flag support, improved logging system with Unix timestamps, and natural-sounding voice generation through advanced chunking;
- **_`2025.01.22`_**: 🔥🔥🔥 **CLI streaming mode supported.** You can now using `--stream` to have fun with stream mode, kudos to [mroigo](https://github.com/mrorigo);
- **_`2025.01.17`_**: 🔥🔥🔥 Style mixing supported! Now, listen the output AMSR effect by simply specific style: `af_sky.4+af_nicole.5`;
- **_`2025.01.15`_**: OpenAI compatible server supported, openai format still under polish!
- **_`2025.01.15`_**: Phonemizer supported! Now `Kokoros` can inference E2E without anyother dependencies! Kudos to [@tstm](https://github.com/tstm);
- **_`2025.01.13`_**: Espeak-ng tokenizer and phonemizer supported! Kudos to [@mindreframer](https://github.com/mindreframer) ;
- **_`2025.01.12`_**: Released `Kokoros`;

## Installation

1. Download the required model and voice data files:

```bash
bash download_all.sh
```

This will download:
- The Kokoro ONNX model (`checkpoints/kokoro-v1.0.onnx`)
- The voices data file (`data/voices-v1.0.bin`)

Alternatively, you can download them separately:
```bash
bash scripts/download_models.sh
bash scripts/download_voices.sh
```

2. Build the project:

```bash
cargo build --release
```

3. (Optional) Install Python dependencies for OpenAI client examples:

```bash
pip install -r scripts/requirements.txt
```

4. (Optional) Install the binary and voice data system-wide:

```bash
bash install.sh
```

This will copy the `koko` binary to `/usr/local/bin` (making it available system-wide as `koko`) and copy the voice data to `$HOME/.cache/kokoros/`.

## Usage

### View available options

```bash
./target/release/koko -h
```

### Logging Configuration

Configure logging output destination and custom file paths:

```bash
# Console logging only (default)
./target/release/koko --log cli text "Hello world"

# File logging only
./target/release/koko --log file text "Hello world"

# Both console and file logging
./target/release/koko --log all text "Hello world"

# Custom log file path
./target/release/koko --log file --log-file custom/path/kokoros.log text "Hello world"

# Disable logging
./target/release/koko --log none text "Hello world"
```

### Generate speech for some text

```
./target/release/koko text "Hello, this is a TTS test"
```

The generated audio will be saved to `tmp/output.wav` by default. You can customize the save location with the `--output` or `-o` option:

```
./target/release/koko text "I hope you're having a great day today!" --output greeting.wav
```

### Generate speech for each line in a file

```
./target/release/koko file poem.txt
```

For a file with 3 lines of text, by default, speech audio files `tmp/output_0.wav`, `tmp/output_1.wav`, `tmp/output_2.wav` will be outputted. You can customize the save location with the `--output` or `-o` option, using `{line}` as the line number:

```
./target/release/koko file lyrics.txt -o "song/lyric_{line}.wav"
```

### Parallel Processing Configuration

Configure parallel TTS instances for the OpenAI-compatible server based on your performance preference:

```
# Best 0.5-2 seconds time-to-first-audio (lowest latency)
./target/release/koko openai --instances 1

# Balanced performance (default, 2 instances, usually best throughput for CPU processing)
./target/release/koko openai

# Best total processing time (optimal at 4 instances on Mac M2)
./target/release/koko openai --instances 4
```

**How to determine the optimal number of instances for your system configuration?**
Choose your configuration based on use case:
- Single instance for real-time applications requiring immediate audio response irrespective of system configuration.
- Multiple instances for batch processing where total completion time matters more than initial latency.
  - This was benchmarked on a Mac M2 with 8 cores and 24GB RAM.
  - Tested with the message:
    > Welcome to our comprehensive technology demonstration session. Today we will explore advanced parallel processing systems thoroughly. These systems utilize multiple computational instances simultaneously for efficiency. Each instance processes different segments concurrently without interference. The coordination between instances ensures seamless output delivery consistently. Modern algorithms optimize resource utilization effectively across all components. Performance improvements are measurable and significant in real scenarios. Quality assurance validates each processing stage thoroughly before deployment. Integration testing confirms system reliability consistently under various conditions. User experience remains smooth throughout operation regardless of complexity. Advanced monitoring tracks system performance metrics continuously during execution.
  - Benchmark results (avg of 3)
    | No. of instances | TTFA | Total time |
    |------------------|------|------------|
    | 1                | 1.87s | 25.1s     |
    | 2                | 2.15s | 16.0s     |
    | 4                | 3.56s | 13.7s     |
    | 8                | 7.73s | 14.7s     |
  - If you have a CPU, memory bandwidth will be the usual bottleneck. You will have to experiment to find a sweet spot of number of instances giving you optimal throughput on your system configuration.
  - If you have a NVIDIA GPU, you can try increasing the number of instances. You are expected to further improve throughput.
  - Note: 8 instances shows diminishing returns due to minimum intra-op threading of 2 threads per instance. Performance may differ if minimum threading is adjusted.
  - Attempts to [make this work on CoreML](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html) face compatibility issues with the Kokoro ONNX model structure, causing fallback to CPU execution. The Kokoro model throws errors about limitations on available nodes, forcing most operations to fall back to CPU processing. This model complexity limitation makes CPU-specific optimization critical for Apple Silicon performance.

*Note: The `--instances` flag is supported in API server mode for parallel processing. CLI commands are integrated with the threading system and use single instance by default for optimal performance (CLI parallel processing support planned for future releases).*

### OpenAI-Compatible Server

1. Start the server:

```bash
./target/release/koko openai
```

2. Make API requests using either curl or Python:

Using curl:

```bash
# Standard audio generation
curl -X POST http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, this is a test of the Kokoro TTS system!",
    "voice": "af_sky"
  }' \
  --output sky-says-hello.wav

# Streaming audio generation (PCM format only)
curl -X POST http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "This is a streaming test with real-time audio generation.",
    "voice": "af_sky",
    "stream": true
  }' \
  --output streaming-audio.pcm

# Live streaming playback (requires ffplay)
curl -s -X POST http://localhost:3000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello streaming world!",
    "voice": "af_sky",
    "stream": true
  }' | \
  ffplay -f s16le -ar 24000 -nodisp -autoexit -loglevel quiet -
```

Using Python:

```bash
python scripts/run_openai.py
```

### Streaming

The `stream` option will start the program, reading for lines of input from stdin and outputting WAV audio to stdout.

Use it in conjunction with piping.

#### Typing manually

```
./target/release/koko stream > live-audio.wav
# Start typing some text to generate speech for and hit enter to submit
# Speech will append to `live-audio.wav` as it is generated
# Hit Ctrl D to exit
```

#### Input from another source

```
echo "Suppose some other program was outputting lines of text" | ./target/release/koko stream > programmatic-audio.wav
```

### With docker

1. Build the image

```bash
docker build -t kokoros .
```

2. Run the image, passing options as described above

```bash
# Basic text to speech
docker run -v ./tmp:/app/tmp kokoros text "Hello from docker!" -o tmp/hello.wav

# An OpenAI server (with appropriately bound port)
docker run -p 3000:3000 kokoros openai
```

## Roadmap

Due to Kokoro actually not finalizing it's ability, this repo will keep tracking the status of Kokoro, and helpfully we can have language support incuding: English, Mandarin, Japanese, German, French etc.

## Copyright

Copyright reserved by Lucas Jin under Apache License.
