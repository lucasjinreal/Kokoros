use ogg::{PacketWriteEndInfo, PacketWriter};
use opus::{Application, Bitrate, Channels, Encoder};
use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn pcm_to_opus_ogg(pcm_data: &[f32], sample_rate: u32) -> Result<Vec<u8>, std::io::Error> {
    // 1. Initialize Opus encoder with Audio application (better for high quality TTS)
    let mut encoder =
        Encoder::new(sample_rate, Channels::Mono, Application::Audio).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Encoder init failed: {:?}", e),
            )
        })?;

    encoder.set_bitrate(Bitrate::Bits(64000)).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Set bitrate failed: {:?}", e),
        )
    })?;

    // Get strict pre-skip value from the encoder
    let pre_skip = encoder.get_lookahead().map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("Get lookahead failed: {:?}", e),
        )
    })? as u16;

    // output buffer
    let mut ogg_buffer = Cursor::new(Vec::new());
    let mut packet_writer = PacketWriter::new(&mut ogg_buffer);

    let serial_no = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or(1);

    // --- 2. Create header packet into OpusHead ---
    let mut id_header = Vec::new();
    id_header.extend_from_slice(b"OpusHead");
    id_header.push(1); // Version
    id_header.push(1); // Channels
    id_header.extend_from_slice(&pre_skip.to_le_bytes()); // Pre-skip (Corrected)
    id_header.extend_from_slice(&sample_rate.to_le_bytes()); // Input Sample Rate
    id_header.extend_from_slice(&0u16.to_le_bytes()); // Gain
    id_header.push(0); // Mapping Family

    packet_writer
        .write_packet(id_header, serial_no, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // --- 3. Create comment header into OpusTags ---
    let comments = vec![("TITLE", "Generated Audio"), ("ENCODER", "Kokoros TTS")];

    let mut comment_header = Vec::new();
    comment_header.extend_from_slice(b"OpusTags");

    let vendor = b"Rust Opus Encoder";
    comment_header.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
    comment_header.extend_from_slice(vendor);

    comment_header.extend_from_slice(&(comments.len() as u32).to_le_bytes());

    for (key, value) in comments {
        let comment_str = format!("{}={}", key, value);
        let comment_bytes = comment_str.as_bytes();
        comment_header.extend_from_slice(&(comment_bytes.len() as u32).to_le_bytes());
        comment_header.extend_from_slice(comment_bytes);
    }

    packet_writer
        .write_packet(comment_header, serial_no, PacketWriteEndInfo::EndPage, 0)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    // --- 4. Encode audio data ---
    let frame_size = (sample_rate as usize * 20) / 1000; // 20ms frames
    // Output buffer recommendation: 4000 bytes is generally enough for max Opus frame
    let mut output_buffer = vec![0u8; 4000];

    let chunks: Vec<&[f32]> = pcm_data.chunks(frame_size).collect();
    let total_chunks = chunks.len();
    let mut samples_processed: u64 = 0; // Track total input samples to avoid drift

    for (i, chunk) in chunks.iter().enumerate() {
        let is_last_chunk = i == total_chunks - 1;

        // Padding for last chunk
        let input_frame = if chunk.len() < frame_size {
            let mut padded = chunk.to_vec();
            padded.resize(frame_size, 0.0);
            std::borrow::Cow::Owned(padded)
        } else {
            std::borrow::Cow::Borrowed(*chunk)
        };

        let encoded_len = encoder
            .encode_float(&input_frame, &mut output_buffer)
            .map_err(|e| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Encoding failed: {:?}", e),
                )
            })?;

        // Calculate Granule Position based on TOTAL processed input samples
        // This avoids floating point accumulation errors.
        // Formula: GP = (Total Input Samples * 48000) / Input Sample Rate
        samples_processed += chunk.len() as u64;

        let granule_pos = (samples_processed * 48000) / sample_rate as u64;

        let end_info = if is_last_chunk {
            PacketWriteEndInfo::EndStream
        } else {
            PacketWriteEndInfo::NormalPacket
        };

        let packet_data = output_buffer[..encoded_len].to_vec();

        packet_writer
            .write_packet(packet_data, serial_no, end_info, granule_pos)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    }

    drop(packet_writer);

    Ok(ogg_buffer.into_inner())
}
