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
    pub fn from_model(model: &str) -> Result<Self, String> {
        if let Ok(model) = OpenAIModel::from_model_name(model) {
            return Ok(API::OpenAI(model));
        }

        if let Ok(model) = AnthropicModel::from_model_name(model) {
            return Ok(API::Anthropic(model));
        }

        if let Ok(model) = GeminiModel::from_model_name(model) {
            return Ok(API::Gemini(model));
        }

        Err(format!("Unknown model: {}", model))
    }

    pub fn from_strings(provider: &str, model: &str) -> Result<Self, String> {
        let api = Self::from_model(model)?;
        let (expected_provider, _) = api.to_strings();

        if expected_provider == provider {
            Ok(api)
        } else {
            Err(format!(
                "Model {} belongs to provider {}, not {}",
                model, expected_provider, provider
            ))
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
