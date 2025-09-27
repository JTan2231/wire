use native_tls::TlsStream;
use std::net::TcpStream;

use crate::types::{Message, MessageBuilder, Tool};

#[async_trait::async_trait]
pub trait Prompt: Send + Sync {
    fn get_auth_token(&self) -> String;

    fn new_message(&self, content: String) -> MessageBuilder;

    fn build_request(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        tools: Option<Vec<Tool>>,
        stream: bool,
    ) -> reqwest::RequestBuilder;

    fn build_request_raw(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
        stream: bool,
    ) -> String;

    /// Ad-hoc prompting for an LLM
    /// Makes zero expectations about the state of the conversation
    /// and returns a tuple of (response message, usage from the prompt)
    async fn prompt(
        &self,
        system_prompt: String,
        chat_history: Vec<Message>,
    ) -> Result<Message, Box<dyn std::error::Error>>;

    async fn prompt_stream(
        &self,
        chat_history: Vec<Message>,
        system_prompt: String,
        tx: tokio::sync::mpsc::Sender<String>,
    ) -> Result<Message, Box<dyn std::error::Error>>;

    async fn prompt_with_tools(
        &self,
        system_prompt: &str,
        chat_history: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<Vec<Message>, Box<dyn std::error::Error>>;

    async fn prompt_with_tools_with_status(
        &self,
        tx: tokio::sync::mpsc::Sender<String>,
        system_prompt: &str,
        chat_history: Vec<Message>,
        tools: Vec<Tool>,
    ) -> Result<Vec<Message>, Box<dyn std::error::Error>>;

    fn read_json_response(
        &self,
        response_json: &serde_json::Value,
    ) -> Result<String, Box<dyn std::error::Error>>;

    async fn process_stream(
        &self,
        stream: TlsStream<TcpStream>,
        tx: &tokio::sync::mpsc::Sender<String>,
    ) -> Result<String, Box<dyn std::error::Error>>;
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "provider", content = "model")]
pub enum API {
    #[serde(rename = "openai")]
    OpenAI(OpenAIModel),
    #[serde(rename = "anthropic")]
    Anthropic(AnthropicModel),
    #[serde(rename = "gemini")]
    Gemini(GeminiModel),
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum OpenAIModel {
    #[serde(rename = "gpt-5")]
    GPT5,
    #[serde(rename = "gpt-4.1")]
    GPT4o,
    #[serde(rename = "gpt-4o-mini")]
    GPT4oMini,
    #[serde(rename = "o1-preview")]
    O1Preview,
    #[serde(rename = "o1-mini")]
    O1Mini,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum AnthropicModel {
    #[serde(rename = "claude-opus-4-1-20250805")]
    ClaudeOpus41,
    #[serde(rename = "claude-opus-4-20250514")]
    ClaudeOpus4,
    #[serde(rename = "claude-sonnet-4-20250514")]
    ClaudeSonnet4,
    #[serde(rename = "claude-3-7-sonnet-20250219")]
    Claude37Sonnet,
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Claude35SonnetNew,
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Claude35Haiku,
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Claude35SonnetOld,
    #[serde(rename = "claude-3-haiku-20240307")]
    Claude3Haiku,
    #[serde(rename = "claude-3-opus-20240229")]
    Claude3Opus,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum GeminiModel {
    #[serde(rename = "gemini-2.5-flash-preview-04-17")]
    Gemini25ProExp,
    #[serde(rename = "gemini-2.0-flash")]
    Gemini20Flash,
    #[serde(rename = "gemini-2.0-flash-lite")]
    Gemini20FlashLite,
    #[serde(rename = "gemini-embedding-exp")]
    GeminiEmbedding,
}

impl API {
    pub fn from_strings(provider: &str, model: &str) -> Result<Self, String> {
        match provider {
            "openai" => {
                let model = match model {
                    "gpt-5" => OpenAIModel::GPT5,
                    "gpt-4o" => OpenAIModel::GPT4o,
                    "gpt-4o-mini" => OpenAIModel::GPT4oMini,
                    "o1-preview" => OpenAIModel::O1Preview,
                    "o1-mini" => OpenAIModel::O1Mini,
                    _ => return Err(format!("Unknown OpenAI model: {}", model)),
                };
                Ok(API::OpenAI(model))
            }
            "anthropic" => {
                let model = match model {
                    "claude-opus-4-1-20250805" => AnthropicModel::ClaudeOpus41,
                    "claude-opus-4-20250514" => AnthropicModel::ClaudeOpus4,
                    "claude-sonnet-4-20250514" => AnthropicModel::ClaudeSonnet4,
                    "claude-3-7-sonnet-20250219" => AnthropicModel::Claude37Sonnet,
                    "claude-3-5-sonnet-20241022" => AnthropicModel::Claude35SonnetNew,
                    "claude-3-5-haiku-20241022" => AnthropicModel::Claude35Haiku,
                    "claude-3-5-sonnet-20240620" => AnthropicModel::Claude35SonnetOld,
                    "claude-3-haiku-20240307" => AnthropicModel::Claude3Haiku,
                    "claude-3-opus-20240229" => AnthropicModel::Claude3Opus,
                    _ => return Err(format!("Unknown Anthropic model: {}", model)),
                };
                Ok(API::Anthropic(model))
            }
            "gemini" => {
                let model = match model {
                    "gemini-2.5-flash-preview-04-17" => GeminiModel::Gemini25ProExp,
                    "gemini-2.0-flash" => GeminiModel::Gemini20Flash,
                    "gemini-2.0-flash-lite" => GeminiModel::Gemini20FlashLite,
                    "gemini-embedding-exp" => GeminiModel::GeminiEmbedding,
                    _ => return Err(format!("Unknown Gemini model: {}", model)),
                };
                Ok(API::Gemini(model))
            }
            _ => Err(format!("Unknown provider: {}", provider)),
        }
    }

    pub fn to_strings(&self) -> (String, String) {
        match self {
            API::OpenAI(model) => model.to_strings(),
            API::Anthropic(model) => model.to_strings(),
            API::Gemini(model) => model.to_strings(),
        }
    }
}
