mod network;
mod tiktoken;
pub mod types;

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
    metrics: HashMap<API, Usage>,
    local_url: Option<String>,
}

// TODO: Actually properly pass error messages up
impl Wire {
    /// Create a new wire
    /// Parameters:
    /// - `tokenizer_path` -- Filepath to an existing tokenizer file
    /// - `download`       -- Optional boolean deciding whether to download a tokenizer file (OAI 400k by default)
    ///                       if it doesn't already exist
    /// - `local_url`      -- Optional URL pointing to a custom endpoint matching the OpenAI API
    ///                       specification. It _must_ match the pattern of
    ///                       `<protocol>://<address>:<port>`
    pub async fn new(local_url: Option<String>) -> Result<Self, std::io::Error> {
        Ok(Self {
            metrics: HashMap::new(),
            local_url,
        })
    }

    pub async fn prompt(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        // TODO: error handling here could probably be a bit more fleshed out
        let (response, usage_delta) = if let Some(url) = &self.local_url {
            let without_protocol = url.split("://").nth(1).unwrap_or(url);

            let parts: Vec<&str> = without_protocol.split(':').collect();
            let host = parts[0];
            let port = parts
                .get(1)
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(80);

            match network::prompt_local(host, port, api.clone(), system_prompt, chat_history).await
            {
                Ok(r) => (r.0, r.1),
                Err(e) => {
                    println!("error prompting LLM: {}", e);
                    return Err(e);
                }
            }
        } else {
            match network::prompt(api.clone(), system_prompt, chat_history).await {
                Ok(r) => (r.0, r.1),
                Err(e) => {
                    println!("error prompting LLM: {}", e);
                    return Err(e);
                }
            }
        };

        let usage = self.metrics.entry(api).or_insert(Usage::new());
        usage.add(usage_delta);

        Ok(response)
    }

    // TODO: Implement streaming
}
