use std::collections::HashMap;

fn get_vocab() -> HashMap<char, usize> {
    let pad = "$";
    let punctuation = r#";:,.!?¡¿—…"«»"" "#;
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    let symbols: String = [pad, punctuation, letters, letters_ipa].concat();
    
    symbols
        .chars()
        .enumerate()
        .collect::<HashMap<_, _>>()
        .into_iter()
        .map(|(idx, c)| (c, idx))
        .collect()
}

fn main() {
    let vocab = get_vocab();
    
    println!("=== TESTING UNICODE CHARACTER SUPPORT ===");
    
    // Test French text
    let french_text = "français";
    let french_phonemes = " fʁɑ̃sˈɛ"; // From espeak
    
    println!("\nFrench text: {}", french_text);
    println!("French phonemes: {}", french_phonemes);
    
    let mut found_chars = Vec::new();
    let mut missing_chars = Vec::new();
    
    for c in french_phonemes.chars() {
        if vocab.contains_key(&c) {
            found_chars.push(c);
        } else {
            missing_chars.push(c);
        }
    }
    
    println!("Found in vocab: {:?}", found_chars);
    println!("Missing from vocab: {:?}", missing_chars);
    
    // Test German text
    let german_text = "Müller";
    let german_phonemes = " mˈʏlɜ"; // From espeak
    
    println!("\nGerman text: {}", german_text);
    println!("German phonemes: {}", german_phonemes);
    
    let mut found_chars = Vec::new();
    let mut missing_chars = Vec::new();
    
    for c in german_phonemes.chars() {
        if vocab.contains_key(&c) {
            found_chars.push(c);
        } else {
            missing_chars.push(c);
        }
    }
    
    println!("Found in vocab: {:?}", found_chars);
    println!("Missing from vocab: {:?}", missing_chars);
    
    // Test filtering behavior
    println!("\n=== TESTING FILTERING BEHAVIOR ===");
    
    let filtered_french: String = french_phonemes.chars().filter(|&c| vocab.contains_key(&c)).collect();
    let filtered_german: String = german_phonemes.chars().filter(|&c| vocab.contains_key(&c)).collect();
    
    println!("French phonemes after filtering: '{}'", filtered_french);
    println!("German phonemes after filtering: '{}'", filtered_german);
}