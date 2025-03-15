mod network;
mod tiktoken;
pub mod types;

use scribe::{error, Logger};
use std::collections::HashMap;

use crate::types::{Message, Usage, API};

// TODO: there probably needs to be a better determination
//       on whether the tokenizer is needed to begin with
//       honestly just need to take a better look at the API response
//
// TODO: dedicated wires? dedicated anonymous wires?

/// LLM interaction API with metric tracking
/// Metrics are ignored if:
/// - Tokenizer mapping path is invalid
/// - There is a tokenizer mapping error during runtime
///
/// Metrics are tracked per API.
pub struct Wire {
    tokenizer: tiktoken::Tokenizer,
    metrics: HashMap<API, Usage>,
}

impl Wire {
    pub async fn new(
        tokenizer_path: Option<std::path::PathBuf>,
        download: Option<bool>,
    ) -> Result<Self, std::io::Error> {
        let tokenizer = match tiktoken::Tokenizer::new(tokenizer_path.clone(), download).await {
            Ok(t) => t,
            Err(e) => {
                panic!("error reading tokenizer file {:?}: {}", tokenizer_path, e);
            }
        };

        Ok(Self {
            tokenizer,
            metrics: HashMap::new(),
        })
    }

    pub async fn prompt(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        // TODO: error handling here could probably be a bit more fleshed out
        let (response, usage_delta) =
            match network::prompt(api.clone(), system_prompt, chat_history).await {
                Ok(r) => (r.0, r.1),
                Err(e) => {
                    error!("error prompting LLM: {}", e);
                    return Err(e);
                }
            };

        let usage = self.metrics.entry(api).or_insert(Usage::new());
        usage.add(usage_delta);

        Ok(response)
    }

    pub fn count_tokens(&self, message: &str) -> usize {
        self.tokenizer.encode(message).len()
    }

    // TODO: Implement
    // pub fn prompt_stream(
    //     &mut self,
    //     api: API,
    //     system_prompt: &str,
    //     chat_history: &Vec<Message>,
    //     tx: std::sync::mpsc::Sender<String>,
    // ) -> Result<Message, std::io::Error> {
    //     // TODO: remarkably stupid gymnastics with this tokenizer
    //     let (response, usage_delta) = match network::prompt_stream(
    //         api.clone(),
    //         chat_history,
    //         system_prompt,
    //         &self.tokenizer,
    //         tx,
    //     ) {
    //         Ok(r) => (r.0, r.1),
    //         Err(e) => {
    //             error!("error prompting LLM: {}", e);
    //             return Err(e);
    //         }
    //     };
    //
    //     let usage = self.metrics.entry(api).or_insert(Usage::new());
    //     usage.add(usage_delta);
    //
    //     Ok(response)
    // }
}
