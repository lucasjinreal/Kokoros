use crate::tts::normalize;
use crate::tts::vocab::VOCAB;
use lazy_static::lazy_static;
use regex::Regex;

lazy_static! {
    static ref PHONEME_PATTERNS: Regex = Regex::new(r"(?<=[a-zɹː])(?=hˈʌndɹɪd)").unwrap();
    static ref Z_PATTERN: Regex = Regex::new(r#" z(?=[;:,.!?¡¿—…"«»"" ]|$)"#).unwrap();
    static ref NINETY_PATTERN: Regex = Regex::new(r"(?<=nˈaɪn)ti(?!ː)").unwrap();
}

// Placeholder for the EspeakBackend struct
#[allow(dead_code)]
struct EspeakBackend {
    language: String,
    preserve_punctuation: bool,
    with_stress: bool,
}

impl EspeakBackend {
    fn new(language: &str, preserve_punctuation: bool, with_stress: bool) -> Self {
        EspeakBackend {
            language: language.to_string(),
            preserve_punctuation,
            with_stress,
        }
    }

    fn phonemize(&self, _text: &[String]) -> Option<Vec<String>> {
        // Implementation would go here
        // This is where you'd integrate with actual espeak bindings
        todo!("Implement actual phonemization")
    }
}

pub struct Phonemizer {
    lang: String,
    backend: EspeakBackend,
}

impl Phonemizer {
    pub fn new(lang: &str) -> Self {
        let backend = match lang {
            "a" => EspeakBackend::new("en-us", true, true),
            "b" => EspeakBackend::new("en-gb", true, true),
            _ => panic!("Unsupported language"),
        };

        Phonemizer {
            lang: lang.to_string(),
            backend,
        }
    }

    pub fn phonemize(&self, text: &str, normalize: bool) -> String {
        let text = if normalize {
            normalize::normalize_text(text)
        } else {
            text.to_string()
        };

        // Assume phonemize returns Option<String>
        let mut ps = match self.backend.phonemize(&[text]) {
            Some(phonemes) => phonemes[0].clone(),
            None => String::new(),
        };

        // Apply kokoro-specific replacements
        ps = ps
            .replace("kəkˈoːɹoʊ", "kˈoʊkəɹoʊ")
            .replace("kəkˈɔːɹəʊ", "kˈəʊkəɹəʊ");

        // Apply character replacements
        ps = ps
            .replace("ʲ", "j")
            .replace("r", "ɹ")
            .replace("x", "k")
            .replace("ɬ", "l");

        // Apply regex patterns
        ps = PHONEME_PATTERNS.replace_all(&ps, " ").to_string();
        ps = Z_PATTERN.replace_all(&ps, "z").to_string();

        if self.lang == "a" {
            ps = NINETY_PATTERN.replace_all(&ps, "di").to_string();
        }

        // Filter characters present in vocabulary
        ps = ps.chars().filter(|&c| VOCAB.contains_key(&c)).collect();

        ps.trim().to_string()
    }
}
