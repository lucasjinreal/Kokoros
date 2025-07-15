use std::borrow::Cow;

use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ort::{
    session::{Session, SessionInputValue, SessionInputs, SessionOutputs},
    value::{Tensor, Value},
};

use super::ort_base;
use ort_base::OrtBase;
use crate::utils::debug::format_debug_prefix;

pub struct OrtKoko {
    sess: Option<Session>,
}
impl ort_base::OrtBase for OrtKoko {
    fn set_sess(&mut self, sess: Session) {
        self.sess = Some(sess);
    }

    fn sess(&self) -> Option<&Session> {
        self.sess.as_ref()
    }
}
impl OrtKoko {
    pub fn new(model_path: String) -> Result<Self, String> {
        Self::new_with_instances(model_path, 1)
    }

    pub fn new_with_instances(model_path: String, total_instances: usize) -> Result<Self, String> {
        let mut instance = OrtKoko { sess: None };
        instance.load_model_with_instances(model_path, total_instances)?;
        Ok(instance)
    }

    pub fn infer(
        &mut self,
        tokens: Vec<Vec<i64>>,
        styles: Vec<Vec<f32>>,
        speed: f32,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
    ) -> Result<ArrayBase<OwnedRepr<f32>, IxDyn>, Box<dyn std::error::Error>> {

        let shape = [tokens.len(), tokens[0].len()];
        let tokens_flat: Vec<i64> = tokens.into_iter().flatten().collect();

        let debug_prefix = format_debug_prefix(request_id, instance_id);
        let chunk_info = chunk_number.map(|n| format!("Chunk: {}, ", n)).unwrap_or_default();
        tracing::debug!("{} {}inference input: tokens_shape={:?}, tokens_count={}, styles_shape={:?}", debug_prefix, chunk_info, shape, tokens_flat.len(), [styles.len(), styles[0].len()]);
        let tokens = Tensor::from_array((shape, tokens_flat))?;
        let tokens_value: SessionInputValue = SessionInputValue::Owned(Value::from(tokens));

        let shape_style = [styles.len(), styles[0].len()];
        let style_flat: Vec<f32> = styles.into_iter().flatten().collect();
        let style = Tensor::from_array((shape_style, style_flat))?;
        let style_value: SessionInputValue = SessionInputValue::Owned(Value::from(style));

        let speed = vec![speed; 1];
        let speed = Tensor::from_array(([1], speed))?;
        let speed_value: SessionInputValue = SessionInputValue::Owned(Value::from(speed));

        let inputs: Vec<(Cow<str>, SessionInputValue)> = vec![
            (Cow::Borrowed("tokens"), tokens_value),
            (Cow::Borrowed("style"), style_value),
            (Cow::Borrowed("speed"), speed_value),
        ];

        if let Some(sess) = &mut self.sess {
            let outputs: SessionOutputs = sess.run(SessionInputs::from(inputs))?;
            let (shape, data) = outputs["audio"]
                .try_extract_tensor::<f32>()
                .expect("Failed to extract tensor");

            // Convert Shape and &[f32] to ArrayBase<OwnedRepr<f32>, IxDyn>
            let shape_vec: Vec<usize> = shape.into_iter().map(|&i| i as usize).collect();
            let data_vec: Vec<f32> = data.to_vec();
            let debug_prefix = format_debug_prefix(request_id, instance_id);
            let chunk_info = chunk_number.map(|n| format!("Chunk: {}, ", n)).unwrap_or_default();
            tracing::debug!("{} {}inference output: audio_shape={:?}, sample_count={}", debug_prefix, chunk_info, shape_vec, data_vec.len());
            let output_array = ArrayBase::<OwnedRepr<f32>, IxDyn>::from_shape_vec(shape_vec, data_vec)?;

            Ok(output_array)
        } else {
            Err("Session is not initialized.".into())
        }
    }
}
