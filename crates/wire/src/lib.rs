mod network;
mod network_common;

pub mod types;

pub mod anthropic;
pub mod api;
pub mod config;
pub mod gemini;
pub mod mock;
pub mod openai;

use crate::config::ClientOptions;
use api::{Prompt, API};
use types::{Message, Tool};

/// Create a client using provider and model identifiers with default options.
///
/// # Errors
/// Returns an error when the provider/model pair is unknown.
pub fn new_client(provider: &str, model: &str) -> Result<Box<dyn Prompt>, String> {
    new_client_internal(provider, model, None)
}

/// Create a client using provider/model identifiers and custom transport options.
///
/// # Errors
/// Returns an error when the provider/model pair is unknown.
pub fn new_client_with_options(
    provider: &str,
    model: &str,
    options: ClientOptions,
) -> Result<Box<dyn Prompt>, String> {
    new_client_internal(provider, model, Some(options))
}

fn new_client_internal(
    provider: &str,
    model: &str,
    options: Option<ClientOptions>,
) -> Result<Box<dyn Prompt>, String> {
    let api = API::from_strings(provider, model)?;

    Ok(match (api, options) {
        (API::OpenAI(model), Some(options)) => {
            Box::new(openai::OpenAIClient::with_options(model, options))
        }
        (API::OpenAI(model), None) => Box::new(openai::OpenAIClient::new(model)),
        (API::Anthropic(model), Some(options)) => {
            Box::new(anthropic::AnthropicClient::with_options(model, options))
        }
        (API::Anthropic(model), None) => Box::new(anthropic::AnthropicClient::new(model)),
        (API::Gemini(model), Some(options)) => {
            Box::new(gemini::GeminiClient::with_options(model, options))
        }
        (API::Gemini(model), None) => Box::new(gemini::GeminiClient::new(model)),
    })
}

pub mod prelude {
    pub use crate::types::{MessageBuilder, MessageWithTools, Tool, ToolWrapper};
    pub use wire_macros::{get_tool, tool};
}

// TODO: These need deprecated in favor of the traits

pub async fn prompt_stream(
    api: API,
    system_prompt: &str,
    chat_history: &Vec<Message>,
    tx: tokio::sync::mpsc::Sender<String>,
) -> Result<Message, Box<dyn std::error::Error>> {
    let response = match network::prompt_stream(api.clone(), chat_history, system_prompt, tx).await
    {
        Ok(r) => r,
        Err(e) => {
            println!("ERROR: {}", e);
            return Err(e);
        }
    };

    Ok(response)
}

// TODO: This should probably return a full response or something similarly useful
pub async fn prompt_with_tools(
    api: API,
    system_prompt: &str,
    chat_history: Vec<Message>,
    tools: Vec<Tool>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let response = match (api, chat_history, tools) {
        (API::OpenAI(model), chat_history, tools) => {
            let client = openai::OpenAIClient::new(model.clone());
            client
                .prompt_with_tools(system_prompt, chat_history, tools)
                .await
        }
        (API::Anthropic(model), chat_history, tools) => {
            let client = anthropic::AnthropicClient::new(model.clone());
            client
                .prompt_with_tools(system_prompt, chat_history, tools)
                .await
        }
        (API::Gemini(model), chat_history, tools) => {
            let client = gemini::GeminiClient::new(model.clone());
            client
                .prompt_with_tools(system_prompt, chat_history, tools)
                .await
        }
    };

    match response {
        Ok(r) => Ok(r),
        Err(e) => {
            println!("error prompting LLM: {}", e);
            Err(e)
        }
    }
}

pub async fn prompt_with_tools_and_status(
    tx: tokio::sync::mpsc::Sender<String>,
    api: API,
    system_prompt: &str,
    chat_history: Vec<Message>,
    tools: Vec<Tool>,
) -> Result<Vec<Message>, Box<dyn std::error::Error>> {
    let response = match (api, chat_history, tools, tx) {
        (API::OpenAI(model), chat_history, tools, tx) => {
            let client = openai::OpenAIClient::new(model.clone());
            client
                .prompt_with_tools_with_status(tx, system_prompt, chat_history, tools)
                .await
        }
        (API::Anthropic(model), chat_history, tools, tx) => {
            let client = anthropic::AnthropicClient::new(model.clone());
            client
                .prompt_with_tools_with_status(tx, system_prompt, chat_history, tools)
                .await
        }
        (API::Gemini(model), chat_history, tools, tx) => {
            let client = gemini::GeminiClient::new(model.clone());
            client
                .prompt_with_tools_with_status(tx, system_prompt, chat_history, tools)
                .await
        }
    };

    match response {
        Ok(r) => Ok(r),
        Err(e) => {
            println!("error prompting LLM: {}", e);
            Err(e)
        }
    }
}
