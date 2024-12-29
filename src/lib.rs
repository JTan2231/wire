mod network;
mod tiktoken;
mod types;

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
/// If you are looking for an API without usage tracking,
/// see `AnonymousWire`.
pub struct Wire {
    tokenizer: Option<tiktoken::Tokenizer>,
    metrics: HashMap<API, Usage>,
}

impl Wire {
    pub fn new(tokenizer_path: Option<std::path::PathBuf>) -> Result<Self, std::io::Error> {
        let tokenizer = match tokenizer_path {
            Some(tp) => match tiktoken::Tokenizer::new(&tp) {
                Ok(t) => Some(t),
                Err(e) => {
                    error!("error reading tokenizer file {:?}: {}", tp, e);
                    error!("tokenization will be ignored");

                    None
                }
            },
            _ => None,
        };

        Ok(Self {
            tokenizer,
            metrics: HashMap::new(),
        })
    }

    pub fn prompt(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
    ) -> Result<Message, std::io::Error> {
        // TODO: error handling here could probably be a bit more fleshed out
        let (response, usage_delta) =
            match network::prompt(api.clone(), system_prompt, chat_history) {
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

    pub fn prompt_stream(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
        tx: std::sync::mpsc::Sender<String>,
    ) -> Result<Message, std::io::Error> {
        // TODO: remarkably stupid gymnastics with this tokenizer
        let tokenizer = match &self.tokenizer {
            Some(t) => Some(t),
            _ => None,
        };

        let (response, usage_delta) =
            match network::prompt_stream(api.clone(), chat_history, system_prompt, tokenizer, tx) {
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
}

/// LLM interaction API without usage tracking.
pub struct AnonymousWire {}

impl AnonymousWire {
    pub fn new() -> Self {
        Self {}
    }

    pub fn prompt(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
    ) -> Result<Message, std::io::Error> {
        // TODO: error handling here could probably be a bit more fleshed out
        let (response, _) = match network::prompt(api, system_prompt, chat_history) {
            Ok(r) => (r.0, r.1),
            Err(e) => {
                error!("error prompting LLM: {}", e);
                return Err(e);
            }
        };

        Ok(response)
    }

    pub fn prompt_stream(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
        tx: std::sync::mpsc::Sender<String>,
    ) -> Result<Message, std::io::Error> {
        let (response, _) = match network::prompt_stream(api, chat_history, system_prompt, None, tx)
        {
            Ok(r) => (r.0, r.1),
            Err(e) => {
                error!("error prompting LLM: {}", e);
                return Err(e);
            }
        };

        Ok(response)
    }
}
