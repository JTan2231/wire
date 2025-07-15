mod network;
mod tiktoken;
pub mod types;

use crate::types::{Message, Tool, API};

pub mod prelude {
    pub use crate::types::{Tool, ToolWrapper};
    pub use wire_macros::{get_tool, tool};
}

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
    local_url: Option<String>,
}

// TODO: Actually properly pass error messages up
impl Wire {
    /// Create a new wire
    /// Parameters:
    /// - `local_url`      -- Optional URL pointing to a custom endpoint matching the OpenAI API
    ///                       specification. It _must_ match the pattern of
    ///                       `<protocol>://<address>:<port>`
    pub async fn new(local_url: Option<String>) -> Result<Self, std::io::Error> {
        Ok(Self { local_url })
    }

    // TODO: Deprecate the tokenizer/usage nonsense
    pub async fn prompt(
        &mut self,
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        // TODO: error handling here could probably be a bit more fleshed out
        let response = if let Some(url) = &self.local_url {
            let without_protocol = url.split("://").nth(1).unwrap_or(url);

            let parts: Vec<&str> = without_protocol.split(':').collect();
            let host = parts[0];
            let port = parts
                .get(1)
                .and_then(|s| s.parse::<u16>().ok())
                .unwrap_or(80);

            match network::prompt_local(host, port, api.clone(), system_prompt, chat_history).await
            {
                Ok(r) => r,
                Err(e) => {
                    println!("error prompting LLM: {}", e);
                    return Err(e);
                }
            }
        } else {
            match network::prompt(api.clone(), system_prompt, chat_history).await {
                Ok(r) => r,
                Err(e) => {
                    println!("error prompting LLM: {}", e);
                    return Err(e);
                }
            }
        };

        Ok(response)
    }

    // TODO: Support for locally hosted models
    pub fn prompt_stream(
        api: API,
        system_prompt: &str,
        chat_history: &Vec<Message>,
        tx: std::sync::mpsc::Sender<String>,
    ) -> Result<Message, Box<dyn std::error::Error>> {
        let response = match network::prompt_stream(api.clone(), chat_history, system_prompt, tx) {
            Ok(r) => r,
            Err(e) => {
                return Err(e);
            }
        };

        Ok(response)
    }
}

// TODO: Filthy workaround
pub fn prompt_stream(
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
    tx: std::sync::mpsc::Sender<String>,
) -> Result<Message, Box<dyn std::error::Error>> {
    let response = match network::prompt_stream(api.clone(), chat_history, system_prompt, tx) {
        Ok(r) => r,
        Err(e) => {
            println!("ERROR: {}", e);
            return Err(e);
        }
    };

    Ok(response)
}

// TODO: Streaming for responses

// TODO: This should probably return a full response or something similarly useful
pub async fn prompt_with_tools(
    api: API,
    system_prompt: &str,
    chat_history: Vec<Message>,
    tools: Vec<Tool>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let response =
        match network::prompt_with_tools(api.clone(), system_prompt, chat_history, tools).await {
            Ok(r) => r,
            Err(e) => {
                println!("error prompting LLM: {}", e);
                return Err(e);
            }
        };

    Ok(response)
}
